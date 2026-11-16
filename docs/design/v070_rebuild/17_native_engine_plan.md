# 17: the native frame engine (Rust) behind protocol version 1

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15), reviewed. Inputs of this revision, kept untracked in the coordinator's speed-work notes: an independent review of revision 0 (2 blockers, 14 major and 15 minor findings, 10 silent decisions), the coordinator's binding decisions on it, and the two measurements of 2026-11-15 (a throwaway Rust pipeline on public photo sets, whose numbers are cited here; a run of the owner's Rust server as it is, described by mechanism only). Section 10 traces every finding and decision to its place here |
| Code base read | `int/b4-stateless` at `7fb83e660` (the crop encoder of plan 13b lane C1 merged): `frame_engine/`, `src/clients/frame_engine.py`, `src/domain/frame_engine_wire.py`, `tests/frame_engine_tests/`, `tests/gpu/`, `docker/frame-engine/`, the compose files, `scripts/bench/gpu_prep_parity.py`, `scripts/datasets/` |
| Reads with | [14](14_roadmap.md) section 4 (how a port is judged), [13](13_backend_speed_plan.md) D1 and the protocol, [13b](13b_crop_vectors_plan.md) sections 6 and 7 (the crop encoder), [the serving trade study](../../research/v060_serving_backend_trade_study.md) sections 5 and 10, [the desktop direction](../photo_library/50_desktop_direction.md) |
| Replaces | track T1 of plan 13 (raised to a build); plan 14 row 7's order ("after phase 3") is overridden by the owner: the native engine starts now |
| Owner decisions in force | 2026-11-15: no further speed work goes into the Python engine; it is frozen as the reference for correct outputs and deleted at the switch (N3); two engines are never kept. Rust (C or C++ where a library needs it). No bias toward any one design: measure, then decide. The engine code is public. Nothing is tuned to this host: its CPU (two Xeon E5-2680 v3, 24 physical cores) is the baseline, thread counts derive from cores and card, and every result names the CPU |
| Not in scope | the API is not in this plan (the owner expects the whole backend to end in Rust; the API port is decided after a running capacity measurement); jobs, storage and training are not either; the model server stays for N0 to N3 (section 4, decision 4.1) |
| Numbers | every number names its set and run. Rows marked "pipeline" are the throwaway Rust pipeline of 2026-11-15 (public sets: COCO 1,000 photos, 120 Commons photos at 18 to 24.5 MP), measured in the same exclusive holds as the Python engine path; other rows cite their plan. Set P (a private set of real-world 20 MP vehicle photos) is a normal part of every speed test; its numbers may be published (owner, 2026-11-15), its source, file names and images stay private |

## 0. The user's job and the measured truth

A person adds photos. Each photo is decoded once on the GPU; the detector, the frame encoder, the
crop encoder (one call per chunk of detected objects), faces, text and thumbnails all run from that
one frame; the person labels by cluster and trains a model. The measure is photos per second from
"photo in" to "stored and findable". The frame engine is the process that touches pixels. Its
Python implementation is bound by the interpreter lock: with the crop cost taken away it stops near
70 photos/s on small photos and 47 on 20 MP photos with the card half idle (plan 13's pooled-crops
row), whatever the card can do.

### 0.1 What bounds the rate on one card (pipeline, one RTX A6000 on PCIe 3.0 x16, 32 in flight)

| Row | Rust pipeline (two holds) | Python engine path (same holds) | What bounds it |
|---|---|---|---|
| Input only, COCO (decode, letterbox, frame square, no models) | 10,155 / 10,427 photos/s | 655 / 622 | Rust: host cores, about 430 photos/s per thread; Python: the interpreter lock |
| Input only, 20 MP | 226 / 232; best 255.5 at 24 decode threads (= the physical cores), 222 at 32, 209 at 48 | 178 / 171 | nvJPEG's host entropy phase on host cores (no JPEG engine on GA102: `nvjpegDecoderCreate` with the hardware backend answers status 7) |
| Detector + frame encoder, COCO | 192 / 189; 197 at 128 in flight | 187 / 184 | the models: their alone row (Z) is 192 to 197 in the same holds |
| Detector + frame encoder, 20 MP | 127 / 124; 138 at 128 in flight | 123 / 119 | the card, shared by the models (about 5 ms per photo) and the decode's device work (about 3 ms) |
| Host CPU per photo, end to end | COCO 7.3 to 7.7 ms; 20 MP 92 to 103 ms | COCO 14.4 to 14.7; 20 MP 106 to 108 | at 20 MP nvJPEG's host phase, the same C code in both |
| Model-server client CPU per photo (both calls) | 0.5 ms | 2.8 to 3.0 ms | |
| Correctness, 50 COCO + 20 Commons photos | model inputs bit-identical (0 differing values); 1,281 resampling weight sets byte-identical; 306 of 306 boxes identical; vectors cosine at least 0.99998 | reference | |

Read plainly: on one card with the frame plan, both engines sit at the card's ceiling, so the
end-to-end rate cannot tell a correct native engine from the Python one. The engine is therefore
judged two ways: by **engine capacity** (the same transport with stand-in models that cost the card
nothing, section 5 N0), and by **reaching the card's ceiling** on each plan. The end-to-end rate of a
plan is stated as the smaller of the two, with both shown. With crops at 224 (plan 13b section 7)
the Python engine is card-bound on COCO (40 photos/s, card 89 to 92% busy) and lock-bound on the 20
MP set (47 to 54, card 78 to 84%): the ingest plan at 20 MP is where the native engine first shows
on the end-to-end rate, and every later cut of card time (section 0a) shows only through it.

### 0.2 The bars (stated once; N0's go and N3's adoption refer to them by id)

"Spread" is the difference between two control runs of the Python engine in the same hold. "Hold"
is one exclusive window on the card in which every arm of a comparison runs, alternating.

| Id | Measure | Bar |
|---|---|---|
| G1 | Card ceiling, per plan (frame plan F; ingest plan C), COCO and the 20 MP set, best of 32, 64 and 128 in flight | native photos/s at least 0.95 of the card ceiling C of the same hold, and not below the Python engine by more than the spread. C = 1000 / (1000 / Z + d): Z is the models-alone row of the plan (photos/s, the same model calls with inputs already resident), d the device ms per photo of the engine's own decode and kernels (card busy share x 1000 / rate in the native input-only row of the same hold). On COCO d is about 0.07 ms and C is Z; on 20 MP d is about 3 ms and Z alone is out of reach on one card (section 10, decision 6a) |
| G2 | Engine capacity: stand-in models (section 5 N0), plans F and C, COCO and the 20 MP set, 32, 64 and 128 in flight, both engines | best native row at least 3.0x the best Python row on COCO and at least 1.2x on 20 MP; native host CPU per photo at most 0.6x the Python engine's on COCO. A row in which the stand-ins themselves are the bound (their own alone row Z0 under 2x the row) is re-run with more stand-in instances, never judged |
| G3 | Latency, real models, 32 and 64 in flight | p99 not above the Python engine's at the same in-flight count; p50 not above it at equal rate |
| G4 | Outputs | every row of section 1.2 for the branches built |
| G5 | Memory, same settings | device memory idle and peak at 32 in flight not above the Python engine's; host RSS peak not above it; NVML's per-process figure not above the engine's own count (section 2.3) plus the idle context baseline |
| G6 | Image size | below the Python engine's image (`docker image ls`, same hold) |
| G7 | Start time, model server already up | container start to `ModelReady` true not above the Python engine's |
| G8 | Faults | N0: 10 of 10 kills at 32 in flight survived (the model server stays up, logs no CUDA error, returns to its memory baseline, serves the next request); a model-server restart with the engine up re-registers without an engine restart. N2: section 1.3 in full |
| G9 | Evidence | every row recorded under the evidence clause (section 5) |

Host CPU per photo at 20 MP is reported, not barred: its floor is nvJPEG's host phase in both
engines. End-to-end rate per plan = min(G2 capacity row, C), both shown.

## 0a. The card budget

After N3 the end-to-end rate on one card is set by GPU time per photo. The work that lowers it is
not engine work; it belongs to plans 13 and 13b. The native engine is what lets those gains through.

| GPU time per photo | Value | Source |
|---|---|---|
| Decode device work (transfer, IDCT, colour, resamples) | COCO about 0.07 ms; 20 MP about 3 ms | pipeline input-only rows: busy share x 1000 / rate |
| Detector | about 0.8 ms | arithmetic: models alone 5.0 ms per photo (196 to 203 photos/s) minus the frame encoder |
| Frame encoder (336) | about 4.2 ms | plan 13's crop row (card time per encoder pass) |
| Crop encoder (224) | 2.1 to 2.3 ms per crop at full batches | plan 13b section 7 |
| Crops per photo | COCO 4.67 at policy `all` (6.5 in a 200-photo sample); set P 2.7 to 3.55 | plan 13 section 2 |

| Plan | Arithmetic | Measured |
|---|---|---|
| F, COCO | 5.1 ms, about 196/s | pipeline 197 at 128 in flight |
| F, 20 MP | about 8 ms, about 125/s | pipeline 138 at 128 in flight (part of the decode work overlaps) |
| C, COCO | 15.4 to 19.4 ms, 52 to 65/s | Python 40/s at 89 to 92% busy (about 22 ms): N0's model-server counters must explain the gap |
| C, set P | 14.0 to 15.8 ms, 63 to 71/s | Python 47 to 54 (lock-bound) |

The levers, each a measurement with its own bar and none of them engine work: a cheaper crop
encoder (measured and closed: no smaller encoder replaces today's for crops, 13b section 8); both cards (encoders data-parallel, and where the decode's device
work runs: N2's two-card row); model build and batching settings (plan 13 P2). Cores one card needs
at its bound: 20 MP frame plan about 125 photos/s x 85 to 100 ms = 11 to 13 cores (pipeline: 12 to
14 busy); COCO 1.3 to 1.5. Two cards at 20 MP need 24 to 28, more than this host's 24 physical
cores, so two cards here are host-bound on large photos (input-only best 255/s); a host with more
cores lifts that without any change to the engine.

## 1. What stays fixed

### 1.1 Protocol version 1 and every answer the API relies on

The API cannot tell the engines apart. `src/domain/frame_engine_wire.py` stays the one definition;
the native engine reads contracts generated from it (section 3, row "Contract").

- One unary KServe v2 `ModelInfer` per photo to the model `frame_engine`, version `1`; inputs
  `image` (BYTES [1], one element), `plan` (BYTES [1], one JSON string), optional `boxes` (FP64
  [n, 4], zero rows allowed: reprocess sends (0, 4)); every input in `raw_input_contents`; BYTES
  elements as a 4-byte little-endian length and the bytes. Outputs and datatypes of
  `OUTPUT_DATATYPES` (`size` INT32 [2], `letterbox` FP64 [3]), filtered to the request's `outputs`
  when it names any (the filter applies to `status`, `decode_route` and `branch_status` too). The
  response echoes `id` and `model_version` `1`; `ModelMetadata` answers `platform` `frame_engine`
  and the shapes of `front.py` `_SHAPES`. `ServerLive`, `ServerReady`, `ModelReady`,
  `ModelMetadata` are answered; every other KServe call `UNIMPLEMENTED` (so `LoadModel`, which the
  API's pool sends when the engine is not ready under `OP_LOAD_PROFILE=ingest`).
- `ModelReady` is called before every photo under the ingest load profile
  (`src/clients/triton_pool_core.py`, `src/clients/model_residency.py`): it reads one atomic and
  never waits on a pool.
- Check order (`front.py`): unknown model (`NOT_FOUND`), request and plan faults
  (`INVALID_ARGUMENT`, fault name first; the first fault in `_check_dependencies` order wins), not
  ready (`UNAVAILABLE`), in-flight cap (`RESOURCE_EXHAUSTED`).
- Photo answers: `status` `ok` or `corrupt`, `huge`, `over_budget`, `gpu_memory`, `decoder`;
  `decode_route` `gpu`, `cpu_<reason>` (`not_jpeg`, `precision`, `components`, `progressive`,
  `truncated`, `small`) or `none`; `branch_status` one JSON object, an entry per reported branch:
  `ok`, `not_run`, or `error` with `unsupported_branch`, `model_error`, `dependency_failed`,
  `image_failed`; `size` once decoded.

| Status | When | What the API does (`src/clients/frame_engine.py`) |
|---|---|---|
| `RESOURCE_EXHAUSTED` | overload: in-flight cap, frame budget wait, region wait, colour-table build wait; a request above the message limit (tonic answers `OUT_OF_RANGE` by itself; a response layer maps that one status, as gRPC C-core answers today) | retries with jittered backoff until the photo's 120 s deadline |
| `UNAVAILABLE` | not ready, draining, model server unavailable, CUDA context lost | retries 3 times |
| `INTERNAL`, details starting `frame engine: ` | the engine's own failure for this photo | no retry |
| `INTERNAL` without that prefix | lost in transit (stream reset, flow control) | retries 3 times |
| `INVALID_ARGUMENT` | malformed request or plan | no retry |
| `NOT_FOUND` | unknown model | no retry |
| `DEADLINE_EXCEEDED` | still running at `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS` (60 s) | no retry |

Readiness means the model server answers, the region is registered and the context is not lost;
the API probes `ModelReady` up to 3 times, 2 s each. Every `OP_FRAME_ENGINE_*` setting keeps its
name and range and refuses a malformed value at start; defaults are kept except
`OP_FRAME_ENGINE_LANES` (section 2.1). Three settings are decided here (review D4):
`OP_FRAME_ENGINE_WORKERS` keeps its meaning (photos in work at once, past admission; the region
window divides by it); `OP_FRAME_ENGINE_TRANSPORT=body` is kept and built in N1a as a mode of the
model-call transport (inputs copied to the request when CUDA shared memory is unavailable);
`OP_FRAME_ENGINE_SWITCH_INTERVAL_MS` means nothing to the native engine, which refuses to start when
it is set, and is retired for good at N3 (`src/config/retired_env.py`) when the Python engine that
reads it is deleted. Every metric name and label of `frame_engine/stage_metrics.py` is kept,
including the `process_*` series (section 3, "Metrics").

### 1.2 Outputs equal to the Python engine's

What the engine controls is bit-identical at one photo in flight: model input tensors, boxes after
settle, blur sums, geometry. Model outputs that depend on the server's batch composition do not
qualify even for Python against itself: the crop calls of one photo meet in the dynamic batcher, so
one photo in flight is not batch 1 (the host-cost lane: on COCO 197 of 200 frame vectors and 1,258
of 1,302 crop vectors were bit-identical between two runs of the same engine). Those rows pass on a
floor measured as Python against Python in the same hold (both numbers reported).

| Output | At 1 in flight (real models) | At 32 in flight | Test |
|---|---|---|---|
| `status`, `decode_route`, `branch_status`, `size`, `letterbox` | identical | identical | `gpu_prep_parity.py` EXACT (status, size, letterbox); `decode_route`, `branch_status` added in N0 |
| Model input tensors (detector canvas, frame square, crop squares; faces and text lines in N1) | byte-identical | not measured (inputs do not depend on load) | new: capture models (section 5, N0) |
| `det_boxes`, `det_scores`, `det_classes` | identical (the pipeline: max abs difference 0) | share of photos with identical detections not below Python against Python | EXACT `det_classes`; boxes and scores new |
| `blur_sums` | identical integers | identical | EXACT |
| `whole_embedding`, `crop_embeddings` | share bit-identical and minimum cosine, each not below Python against Python | same floor | `gpu_prep_parity.py compare` |
| `crop_embedded`, `face_embedded`, overflow counts | identical | identical | new |
| `crop_jpegs` | same count and pixel sizes; decoded pixels identical; bytes identical (libjpeg-turbo 3.1.2 as OpenCV 5.0.0.93 bundles, quality, 4:2:0, no optimisation, accurate DCT); where a build or setting cannot match, sizes within 0.5% (question Q6) | same | new |
| `derivative_jpegs` (N1c) | decoded pixels within 1 level of the Python engine's; PSNR to Pillow 40 dB stays; bytes identity reported | same | `tests/gpu/test_gpu_renditions.py` |
| Colour (N1a) | profile class identical; table bytes identical (Little CMS 2.19, Pillow 13.0.0's build) on the 88-profile corpus (COCO's 85 plus 3 fixtures) | same | `tests/gpu/test_gpu_colour.py` plus the corpus |
| Host route (N1a) | per format in the corpus: lossless formats identical, JPEG identical (libjpeg-turbo 3.1.2 as simplejpeg 1.9.0 bundles), other lossy formats within 1 level, every refusal the same status | same | new corpus |
| `face_*` (N1b) | `BOX_ATOL_PX` 1.0 and cosine p01 at least 0.995 against the reference path; LFW, CALFW, CPLFW and WIDER FACE AP not below the Python branch's own numbers on the same pairs by more than its two-run difference | same | `tests/gpu/test_gpu_faces.py` |
| `ocr_*` (N1c) | ground truth only: COCO-Text 400 photos strict F1 and spotted words not below the Python branch's on the same photos by more than its two-run difference; the existing checks (substrings, counts, quads in the frame) | same | `tests/gpu/test_gpu_ocr.py` |

Bits against ground truth (review D1): the bit rows are the first gate, cheap and sharp. A bit row
that fails for a reason the lane names (for example a resampling weight one ulp apart) is judged by
the ground-truth rows of plan 13 for that branch, within the Python-against-Python spread; an
unexplained bit difference is a defect. Resampling weights: the pipeline reproduced numpy's pairwise
row sum and Python's round-half-even and matched all 1,281 weight sets of four sets byte for byte;
N0's test covers every (input, output, offset, take) tuple the three N0 sets produce, not a sample.
Lanczos weights (`np.sinc`, renditions) are judged by the rendition row above.

### 1.3 The fault rule

A decode or prep fault must not take the models down. Kept by the same means: a process apart from
the model server; the kill test (killed at least 50 times at 32 in flight in N2; the model server
stays up, logs no CUDA error, returns to its memory baseline, serves the next request); region names
with the fixed prefix `op-fe-` purged at start (the pipeline saw a killed process leave its region
registered); unregister before free at stop.

A lost context refuses work with `UNAVAILABLE`, turns readiness false and exits non-zero for the
restart policy. Lost means one of the codes whose documentation says "the process must be terminated
and relaunched": 226, 700, 702, 710, 714 to 719, 721, 810, 911
(https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html, `cudaError`). The Python
list (`frame_engine/cuda_fault.py`) is the 700-range subset; 214 (uncorrectable ECC) carries no such
sentence and fails only the photo. The engine calls the driver API; a unit test maps each code to its
`CUresult` by name from cudarc's `sys` enum and lists any code without a driver counterpart with the
reason.

A panic in one photo's work fails that photo (`INTERNAL` with the prefix) and the process goes on:
the release profile keeps `panic = "unwind"`; every job on a decode, launch or finish thread runs
under `std::panic::catch_unwind`, so a panic never ends a thread (a lost decode lane would be lost
for good); tokio isolates panics inside tasks only. The photo's device memory is released by its
owner's drop after a stream wait. A test panics inside a lane's job and then serves the next photo
on that lane.

### 1.4 GPU profiles

Every profile of `config_templates/profiles/` keeps working, including a 12 GB card (the RTX 3080
Ti here). The native engine owns plan 13 item E4 (N2): sizes per profile from measured memory, a
frame budget for at least one 50 MP photo, the 12 GB card serving the full plan, the engine's own
accounting asserted against NVML there (section 2.3).

### 1.5 The model server

Triton over CUDA shared memory stays for N0 to N3 (decision 4.1 settles it). The model call sits
behind one interface (`ModelCalls`, section 2.0) with one filling; an in-process runtime is a second
filling only when a measurement reopens 4.1, and the losing filling is then deleted, not kept.

## 2. Architecture

### 2.0 The module boundary

The general part can later become or join an open-source engine; the product part is
OpenProcessor's.

| Crate | Holds | Never holds |
|---|---|---|
| `native/gpu_frame` (general) | decode lanes (nvJPEG bindings, lane pool); the frame pool (device frames from one stream-ordered pool, counted); the region and slab (one exported allocation, its IPC handle, first-fit slices with coalescing); admission and budget (count and bytes, fair, bounded waits); the kernel loader (NVRTC with CuPy-equivalent options, cubin, module cache); the `ModelCalls` trait and its filling for a model server over CUDA shared memory (Triton's protos, region registration and purge) | any type from the plan, the wire, a branch or a product kernel |
| `native/frame_engine` (product) | the kernels (`.cu` sources), the branches and their rounds, the KServe front of protocol version 1, the plan vocabulary and its parser, the geometry and routing ports, settings, metrics | an nvJPEG or CUDA FFI call (it goes through `gpu_frame`) |

The interface, in Rust terms (names fixed in N0's first file of each crate):

| `gpu_frame` item | Shape |
|---|---|
| `Lanes::decode` | encoded bytes (a `Bytes` slice kept alive until the lane's stream was synchronised: nvJPEG keeps pointers into it with `save_stream = 0`) into a `DeviceFrame`, on the lane's stream; returns size, components and the backend used, or a typed decode error |
| `FramePool::alloc` | bytes into a `DeviceBuf` charged to the cap; freed by drop after its last stream was waited for |
| `Admission::admit` | count and bytes with a deadline into a permit, or `Overload(cause)` |
| `Region::slice` | bytes with a wait bound into a `Slice` (offset, length) of the registered region, or `Overload` |
| `Kernels::load` | CUDA C source text plus a name into a module compiled for the current device; `Module::launch` with grid, block, arguments, stream |
| `ModelCalls::infer` | model name, inputs as region slices (name, datatype, shape, slice), output names, deadline: a future of output tensors or a typed call error (unavailable, deadline, model error, region unknown) |

### 2.1 Processes and threads

One OS process per card (instance `fe0`), in its own container, beside the card's model server.

| Role | Execution unit | Count, derived | Does | Never does |
|---|---|---|---|---|
| Network | tokio runtime, multi-thread | from N0's front CPU per photo by role: the smallest count that keeps each thread under half busy at the card ceiling, at least 2 (the pipeline used 4) | the tonic KServe front; the tonic client channels (`OP_FRAME_ENGINE_TRITON_CHANNELS`); one async control task per photo (header markers to the frame header, the end-of-image check, then the reservation); the health tick | pixel work, decode, JPEG encode, blocking waits |
| Decode | OS threads, each owning one nvJPEG lane (decoder states, pinned and device buffers, jpeg stream, CUDA stream) | `OP_FRAME_ENGINE_LANES`; default the physical cores available to the container: distinct (package, core) pairs among the CPUs of `sched_getaffinity`, capped by the cgroup `cpu.max` quota; never hyperthreads (pipeline: knee at 24 = the physical cores) | the parse and host phase, transfer, device decode, orientation, the round-1 resamples into region slices, one stream wait | model calls |
| Launch | OS threads, each with its own CUDA stream | measured in N0 by thread role, then derived from cores (start: physical cores / 6, at least 2) | round-2 kernels (crop squares; face and text warps in N1), blur sums, renditions; one stream wait each | decode |
| Finish | OS threads | `OP_FRAME_ENGINE_ENCODE_THREADS` (default 8, measured in N0) | crop pixels to pinned host memory and JPEG-encoded (libjpeg-turbo), host post-processing, response build | anything that feeds the card |

The tokio documentation decides the split: CPU-bound work goes to a dedicated pool and returns
through a oneshot channel; `spawn_blocking` is not used for decode (its pool grows to 512 threads)
(https://docs.rs/tokio/1.54.1/tokio/index.html, "CPU-bound tasks and blocking code"). Every pool
thread calls cudarc's `CudaContext::bind_to_thread` before its first raw call, since the nvJPEG FFI
does not go through cudarc. Per thread, as the nvJPEG documentation requires: the jpeg stream, the
decoder state, the pinned and device buffers, and on the encoder side the encoder state
(https://docs.nvidia.com/cuda/nvjpeg/index.html, section 1.3); the library handle is thread safe
(section 2.2.13). The decoder handle is shared across lanes as the Python lanes and the pipeline do:
practice, not a documented guarantee. NUMA pinning of decode threads to the card's socket is an N0
and N2 arm, not a default.

### 2.2 Stages, queues and their bounds

```text
gRPC request ──► admission ──► route + budget ──► decode thread ──► round-1 calls ──► launch pool ──► round-2 calls
 (Bytes, no       in-flight     header, EOI,       nvJPEG, orient,   over CUDA         crop squares,    crop chunks
  copy)           cap           bytes reserved     resamples, sync   shared memory     sync             │
 response ◄── finish (outputs, release) ◄── finish pool (crop JPEGs, post-process) ◄─────────────────────┘
```

| Wait | Bound | Answer when exceeded |
|---|---|---|
| Admission | `OP_FRAME_ENGINE_MAX_IN_FLIGHT` photos admitted (running or queued) | `RESOURCE_EXHAUSTED` at once |
| Route and frame budget | the route is decided before any reservation: header markers, then the end-of-image check (a reverse `memchr` from the end), so a truncated file is routed `cpu_truncated` with its own byte need (`host_need_bytes`) before bytes are reserved. `OP_FRAME_ENGINE_BUDGET_MIB`, first come first served (a fair semaphore, permits in KiB), need from `prep_budget.py`'s arithmetic plus the resample scratch (section 2.3) | `RESOURCE_EXHAUSTED` after `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`; larger than the whole budget: `over_budget` |
| In work | `OP_FRAME_ENGINE_WORKERS` photos past admission hold frames and region slices | waits inside the budget's bound |
| Region slices | a photo holds at most its window, `REGION_MIB / WORKERS` (`engine.py`), or one call when that is larger. Round-1 slices are taken by the control task before dispatch; round-2 slices (data-dependent counts) are staged by the launch thread inside the photo's window: when the next chunk does not fit, the photo collects its own oldest call first (as `PhotoCalls`), so a photo never waits on other photos while holding its frame | `RESOURCE_EXHAUSTED` after `OP_FRAME_ENGINE_SLAB_WAIT_SECONDS` (first fit can fragment) |
| Model calls | each call `min(OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS, what is left of the request)` as the `grpc-timeout` header (`tonic::Request::set_timeout`) and a tokio timeout; dropping the future cancels the stream | `UNAVAILABLE` and an immediate health probe; readiness follows the probe, never one call |
| Request | `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS` | `DEADLINE_EXCEEDED`, logged and counted with its stage |

Cancellation: each photo carries a cancelled flag, set when the caller's stream ends or the request
deadline passes; every hand-off (decode queue, launch, finish) checks it and releases the photo's
budget and slices instead of starting work (the Python front cancels a queued photo the same way).

### 2.3 Memory: ownership and accounting

1. The request arrives in tonic's per-message buffer; with prost's `bytes` option every
   `raw_input_contents` element is a reference-counted slice of it (tonic's `copy_to_bytes`
   delegates to `BytesMut`). The slice lives until the lane's stream was synchronised.
2. The frame is one device allocation per photo from a dedicated CUDA memory pool
   (`cuMemPoolCreate`, allocated stream-ordered); its owner is the photo; pool jobs borrow it; it is
   freed after every stream that touched it was waited for (the rule of `engine.py` made a type).
3. Model inputs are written by kernels into slices of one region: a single `cuMemAlloc` made at
   start through cudarc's `result::malloc_sync` (cudarc's `CudaStream::alloc` takes pool memory
   whenever the device supports it, and pool memory cannot be exported by `cuIpcGetMemHandle`),
   exported with `cuIpcGetMemHandle`, registered with the 64-byte handle as `raw_handle`
   (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html). A slice returns only after
   its call ended and its writer's stream was waited for.
4. Answers arrive as gRPC messages; crop pixels come back through pinned staging per finish thread;
   JPEG bytes become `Bytes` in the response without a copy.

The engine's own byte count is the cap (review D6: a hard cap, not a release threshold).
`OP_FRAME_ENGINE_POOL_MIB` caps every device allocation the engine makes except the region; an
allocation past it fails as an out-of-memory error; `CUDA_ERROR_OUT_OF_MEMORY` (2) and nvJPEG status
5 (allocator failure) answer `gpu_memory`, as `prep_budget.is_memory_error` does. The pool's release
threshold only keeps freed memory cached.

| Class | Size | Charged to |
|---|---|---|
| Region | `OP_FRAME_ENGINE_REGION_MIB`, fixed at start | outside the cap, reported at start |
| Lane buffers | grow with the largest photo a lane decoded (pipeline: about 360 MB per lane at 20 MP including its frame and scratch) | the cap: nvJPEG's device allocations go through a counting allocator passed as `nvjpegDevAllocator_t` to `nvjpegCreateEx` (nvJPEG docs, `nvjpegDevAllocator_t`) |
| Frame and its oriented copy | w x h x 3, twice while an EXIF orientation is applied | the frame budget (and so the cap) |
| Resample scratch | rows x width x 12 bytes per model input being written (about 28 MB for a 20 MP detector letterbox) | the frame budget (a term the Python budget leaves out) |
| Weight sets | small; a byte-capped least-recently-used cache (the pipeline's grew without a bound) | the cap |
| Colour tables (N1a) | 48 MiB each, `OP_FRAME_ENGINE_COLOUR_TABLES` | the cap |
| Rendition scratch (N1c) | `prep_budget.py` arithmetic | the frame budget |
| Crop pixel staging | pinned host memory per finish thread, reused | host, not the card |

The GPU tier asserts on the 12 GB profile that NVML's per-process device memory stays within the
context baseline (measured idle) plus the region plus `OP_FRAME_ENGINE_POOL_MIB`.

### 2.4 Channel settings, both directions

| Direction | Setting | Python today | Native | Doc |
|---|---|---|---|---|
| API to engine | idle pings | the API pings every 30 s with no call in flight; the Python server permits it (`SERVER_OPTIONS`) because C-core sends GOAWAY `too_many_pings` otherwise | a tonic server enforces no ping policy: h2 answers every ping, no strike counter (read from h2 0.4 source); proved by the contract test (section 5, N0) | https://grpc.io/docs/guides/keepalive/ |
| API to engine | server keepalive | none sent | `http2_keepalive_interval` left unset | https://docs.rs/tonic/0.14.9/tonic/transport/server/struct.Server.html |
| API to engine | message limit | `OP_FRAME_ENGINE_MAX_MESSAGE_MIB` (256) | `max_decoding_message_size` and `max_encoding_message_size` on the generated server; over the limit tonic answers `OUT_OF_RANGE` (`codec/decode.rs`), mapped to `RESOURCE_EXHAUSTED` | same |
| API to engine | receive windows | C-core defaults; the API's channel asks 1 MiB stream and 2 MiB connection windows for what it receives | `initial_stream_window_size` 1 MiB, `initial_connection_window_size` 2 MiB (tonic defaults are 64 KiB, too small for a 10 MB body); `http2_adaptive_window` is an N0 arm | same |
| Engine to model server | keepalive | 30 s time, 10 s timeout, pings without calls permitted (Triton accepts pings 20 s apart: `--grpc-http2-min-recv-ping-interval-without-data=20000`) | `Endpoint::http2_keep_alive_interval(30 s)`, `keep_alive_timeout(10 s)`, `keep_alive_while_idle(true)` (the pipeline's values) | https://docs.rs/tonic/0.14.9/tonic/transport/channel/struct.Endpoint.html |
| Engine to model server | message limits | 512 MiB both ways | the same on the generated client | same |
| Engine to model server | connections | `OP_FRAME_ENGINE_TRITON_CHANNELS` (4) | 4 endpoints, one HTTP/2 connection each (pipeline: 256 calls in flight, 0 errors) | same |
| Engine to model server | reconnect and a recreated address | backoff 250 ms to 3 s; name re-resolved at most every 1 s | a tonic channel reconnects on the next call after a failure and its connector resolves the name on each connect; the health tick (`OP_FRAME_ENGINE_HEALTH_SECONDS`, 2 s) probes, so a restarted or moved server is found within one tick; `connect_timeout` = `OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS` | the tonic page says nothing of reconnect: N2's restart and new-address rows are the proof |

### 2.5 The checklist (host and card busy together)

| # | Mechanism | Where |
|---|---|---|
| K1 | Admission by count and by decoded bytes read from the header; no unbounded wait | 2.2 |
| K2 | Decode runs ahead continuously and hands over single photos | decode pool |
| K3 | One decode feeds every model input on the device; only the encoded file crosses the bus in | 2.3 |
| K4 | Batches form in the model server from inputs already resident | decision 4.6 |
| K5 | GPU decodes are a bounded pool of lanes, one state and stream each; the count derives from cores | 2.1 |
| K6 | Frames waiting for the models are bounded by the budget | 2.2 |
| K7 | A partial batch leaves on the model server's timer; the mean batch is reported per row | plan 13 P2 |
| K8 | No large allocation per photo on the hot path: persistent region, frame pool, reused staging, cached weights | 2.3 |
| K9 | Per-photo host work off the threads that feed the card | finish pool |
| K10 | One owner per stream, per nvJPEG state, per model session | 2.1 |
| K11 | No process-wide lock on the request path: atomics, per-lane state, a fair semaphore | Rust |

## 3. Building blocks

Versions are the latest on 2026-11-15 and pinned exactly in `native/Cargo.lock`; licences are
recorded at selection (plan 14 section 3) and checked by `cargo deny` (section 3.1).

| Need | Choice | Why, and the alternative | Doc | Licence |
|---|---|---|---|---|
| Toolchain | Rust 1.99.1, edition 2024, `native/rust-toolchain.toml` (components rustfmt, clippy) | owner: Rust preferred; ownership types enforce the stream-wait-before-free rules the Python code keeps by comment | https://rust-lang.github.io/rustup/overrides.html | MIT / Apache-2.0 |
| CUDA driver, streams, modules, NVRTC | `cudarc` 0.19.10, features `std driver nvrtc cuda-13020 dynamic-loading`, raw `result` and `sys` layers where the safe API cannot (IPC, cubin) | maintained; run-time loading; `cust` last released 2022 | https://docs.rs/cudarc | MIT OR Apache-2.0 |
| Kernel build | the CUDA C sources moved once into `native/frame_engine/kernels/*.cu`, read by both engines; NVRTC with exactly CuPy 14.2.0's `RawModule` options: `-ftz=true`, `--device-as-default-execution-space`, `-arch=sm_<cc>` from the device's compute capability, output a cubin via `nvrtcGetCUBIN` (CuPy's rule: compute_<cc> and PTX only when NVRTC does not list the arch, `nvrtcGetSupportedArchs`); loaded with `cuModuleLoadData`. No disk cache in N0 (the pipeline's compile took 0.7 to 1.5 s at start; G7 decides) | cudarc's safe `compile_ptx_with_opts` returns PTX for a driver JIT and leaves `ftz` false: not CuPy's binary. CV-CUDA or NPP: different arithmetic | https://github.com/cupy/cupy/blob/v14.2.0/cupy/cuda/compiler.py ; https://docs.nvidia.com/cuda/nvrtc/index.html | AGPL (this repository) |
| nvJPEG | bindings taken from the owner's Rust server (`src/decode/nvjpeg/ffi.rs` at commit 59fbb42, its nvJPEG part only): symbols resolved at run time with `libloading`, every create paired with its destroy in a `Drop`, status names, a `Send`-not-`Sync` lane context, a SAFETY note per call. Its CUDA runtime calls are not taken: streams and memory come from cudarc (a `CUstream` is a `cudaStream_t`). The decode order (`GPU_HYBRID` then `HYBRID`, RGBI output, interpolated chroma, decoupled phases) equals `nvjpeg_lanes.py` and the pipeline's. Encoder symbols (8) are added in N1c | the pipeline's own bindings (16 functions) link `libnvjpeg` at build time, never destroy a handle, and panic; the server's are tested code that already does what a product needs. Its MIT OR Apache-2.0 notice is kept in the file header and `ATTRIBUTION.md` | https://docs.nvidia.com/cuda/nvjpeg/index.html (1.3, 2.2.1, 2.3.4) | MIT OR Apache-2.0 (the file); nvJPEG under the CUDA EULA |
| CUDA libraries | the NVIDIA wheels the Python image pins: NVRTC 13.2.86, nvJPEG 13.2.3.58, runtime 13.2.86, unpacked in a build stage | the same NVRTC build gives the same cubin, the same nvJPEG the same pixels | `docker/frame-engine/Dockerfile` table | CUDA EULA |
| Host geometry and routing | ports of `prep_geometry.py`, `src/domain/image_geometry.py`, `src/domain/jpeg_markers.py` (the pipeline's `geom.rs` is the start), checked against goldens generated from the Python functions | pure functions; goldens make drift a failing test | | AGPL |
| Contract | `scripts/codegen/export_api_contracts.py` gains the generator `frame-engine`, writing `contracts/json/frame_engine_v1.json` (tensors, datatypes, shapes, statuses and their order, plan faults, branch outputs, decode routes, request validation rules, the ping policy) and `contracts/json/frame_engine_plan_cases.json` (plan texts with their parse result or fault); the `api-contracts-drift` hook's `files:` pattern lists both | the API keeps the Python wire module; the contract is the boundary, generated | [contracts/README.md](../../../contracts/README.md) | AGPL |
| gRPC | `tonic` 0.14.9, `tonic-prost`, `prost` 0.14.6, `protox` 0.10.0 (no `protoc`), over Triton's `grpc_service.proto` and `model_config.proto` vendored at r26.10 (the model server's release) | one proto set for front and client; `triton-client` crates unmaintained | https://docs.rs/tonic/0.14.9 ; https://docs.rs/prost-build/latest/prost_build/struct.Config.html | MIT; Apache-2.0; protos BSD-3 |
| Async and pools | `tokio` 1.54.1 for the network; `std::thread` plus `crossbeam-channel` 0.5.17 for decode, launch, finish | tokio's guidance for CPU-bound work | https://docs.rs/tokio/1.54.1 | MIT; MIT OR Apache-2.0 |
| Host JPEG (crop encode in N0; decode in N1a) | `turbojpeg` 1.5.1 over `turbojpeg-sys` 1.2.0 with `TURBOJPEG_SOURCE=explicit` against libjpeg-turbo 3.1.2 built from its release tarball (SHA-256 pinned) in the image's build stage, static, SIMD required (nasm); decode strict through `tj3Set(TJPARAM_STOPONWARNING, 1)` from `turbojpeg-sys`; accurate DCT and fancy upsampling as simplejpeg; the two warning messages on which the Python route falls back to Pillow fall back the same way | 3.1.2 is the version both simplejpeg 1.9.0 and OpenCV 5.0.0.93 bundle (read from the wheels); the crate vendors 3.1.0. `zune-jpeg`: other IDCT | https://raw.githubusercontent.com/honzasp/rust-turbojpeg/master/turbojpeg-sys/README.md ; https://libjpeg-turbo.org/Documentation/Documentation (TurboJPEG C API, `TJPARAM_STOPONWARNING`) | Unlicense OR MIT; libjpeg-turbo BSD-3 and IJG |
| Other host formats (N1a) | the accepted set equals today's: what `src/utils/image_decode.py` opens through Pillow 13.0.0's plugins, with its mode conversions (`convert('RGB')` for palette, 16-bit, LA, RGBA, CMYK) and `exif_transpose`. One corpus file per accepted plugin and mode; each format is ported in Rust (`image` 0.25.10 where it decodes identically), or bound to the same C library Pillow bundles (OpenJPEG 2.5.4, libavif, libtiff), decided per format by the corpus | the `image` crate alone decodes fewer formats and converts 16-bit and palette data by its own rules | https://docs.rs/image ; https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html | per library, recorded in N1a |
| Colour (N1a) | `lcms2` 6.2.0 over `lcms2-sys` 4.0.7 built from Little CMS 2.19 (Pillow 13.0.0 ships 2.19); ports of `src/utils/icc_srgb.py` and `src/utils/icc_curves.py` (tag table, parametric curves, the colorant and curve thresholds, the matrix-shaper test); perceptual intent, flags 0 as Pillow; the 2^24-entry table built single-flighted on one thread, so the transform is never shared across threads; gathered on the GPU by `colour.cu` | one colour path for both routes | https://docs.rs/lcms2 ; https://www.littlecms.com/ | MIT |
| Text-box offset (N1c) | the same algorithm as today, pyclipper 1.4.0's Clipper 6.4.2 (`JT_ROUND`, `ET_CLOSEDPOLYGON`, its integer conversion, default arc tolerance): either a binding of the Clipper 6.4.2 C++ source (a C++ toolchain in the build stage) or a Rust port, decided by the OCR parity gate; contours and minimum-area rectangles as OpenCV's `findContours` (`RETR_LIST`, `CHAIN_APPROX_SIMPLE`) and `minAreaRect`, ported and checked on the gate's photos | the `clipper2` crate binds Clipper2, not the 6.4.2 behaviour | https://github.com/fonttools/pyclipper ; https://sourceforge.net/projects/polyclipping/ | Boost (Clipper 6.4.2) |
| Face alignment (N1b) | `nalgebra` 0.35.0 for the SVD, determinant, rank and inverse of `faces/alignment.py` | no hand-written SVD | https://docs.rs/nalgebra | Apache-2.0 |
| Metrics | `prometheus-client` 0.26.0 with the names and labels of `stage_metrics.py`; the `process_*` series of Python's `ProcessCollector` (CPU seconds, resident and virtual memory, open and maximum descriptors, start time) read from `/proc/self`; served on `OP_FRAME_ENGINE_METRICS_PORT` by a hyper server (already in the tree through tonic) | the release acceptance greps `op_frame_engine_stage_seconds_total` and `_observations_total` | https://docs.rs/prometheus-client | Apache-2.0 OR MIT |

**The owner's Rust server, and why it is not a dependency.** Measured as it is (2026-11-15), it
confirms the design by mechanism: decode lanes on plain OS threads scale with cores in one process,
admission by bytes read from the header, batches formed only from resident inputs. It cannot serve
this product: it loads detector-shaped models with float inputs only (the frame encoder, with its
uint8 input and vector output, is refused), it decodes again for every prepared input (the frame is
private to one call), it has no crop, face, text or rendition path, and it runs models through ONNX
Runtime's TensorRT provider, not the native plans the model repository holds (trade study section
5.3). Its letterbox is a non-antialiased bilinear; moving the frame encoder's square to that filter
would move the vector space on large photos. So the general parts live in `native/gpu_frame`, AGPL
like the whole repository (owner: licence questions deferred; question Q3 is not a gate), and its
nvJPEG bindings are the starting point above. Code moves into that server only as the copyright
holder's act; outside contributions to `native/gpu_frame` would need their authors' consent for a
relicence (section 8, Q3).

### 3.1 Rust gates (in N0's first commit)

| Gate | Command | Where |
|---|---|---|
| Format | `cargo fmt --manifest-path native/Cargo.toml --all -- --check` | pre-commit hook `cargo-fmt`, `files: ^native/` |
| Lint | `cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets --locked -- -D warnings`; workspace `[lints]` (https://doc.rust-lang.org/cargo/reference/manifest.html#the-lints-section): `unsafe_op_in_unsafe_fn = "deny"`, clippy `undocumented_unsafe_blocks`, `unwrap_used` (outside tests), `cognitive_complexity`, `too_many_lines`, `allow_attributes`, `allow_attributes_without_reason`, `missing_errors_doc`, `missing_panics_doc`, `missing_safety_doc` all `"deny"`, and rustc `missing_docs = "deny"` (https://rust-lang.github.io/rust-clippy/stable/index.html) | hook `cargo-clippy`, `files: ^native/` |
| Complexity | clippy `cognitive_complexity` at 15 (the Python gate's mccabe limit) and `too_many_lines` at 100 (the Python `max-statements` is 60), `too_many_arguments` at its default 7; thresholds in `native/clippy.toml` (https://doc.rust-lang.org/clippy/lint_configuration.html). A function over them is split by concern; there is no baseline and no `#[allow]` | the lint step above |
| Suppressions | every suppression is `#[expect(.., reason = "..")]` (`allow_attributes`, `allow_attributes_without_reason`): it names why and fails when the lint stops firing | the lint step above |
| Documentation | `missing_docs` and the clippy `missing_*_doc` lints above (a public item says what it is for, with the unit, owner or invariant; `# Errors`, `# Panics`, `# Safety` where they apply); `cargo doc --manifest-path native/Cargo.toml --workspace --no-deps --locked` with `RUSTDOCFLAGS="-D warnings"` and `rustdoc::broken_intra_doc_links`, `private_intra_doc_links` denied (https://doc.rust-lang.org/rustdoc/lints.html) | hook `cargo-doc`, `files: ^native/` |
| Duplication | `scripts/codegen/check_duplication.py` (jscpd, 12 lines and 100 tokens, test files not scanned) also scans `native/**/*.rs`; baseline `scripts/codegen/baselines/duplication.txt`, which only shrinks; native has no entry | hook `check-duplication` |
| Tests | `cargo test --manifest-path native/Cargo.toml --workspace --locked` (no GPU, no network) | hook `cargo-test`, `files: ^native/` |
| Licences, bans, sources | `cargo deny --manifest-path native/Cargo.toml check licenses bans sources` with `native/deny.toml`: licences allowed MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0, Unlicense, CC0-1.0, BSL-1.0, MPL-2.0 (each compatible with AGPL-3.0-or-later: https://www.gnu.org/licenses/license-list.html); sources crates.io only (no git dependencies) | hook `cargo-deny`, `files: ^native/` |
| Advisories | `cargo deny ... check advisories` (RustSec database; needs the network) | pre-push stage, and CI |
| CI | one job `native` in `.github/workflows/ci.yml`, run when `native/` changes (the `changes` job): toolchain from `rust-toolchain.toml` through `rustup show`, `actions/cache` for `~/.cargo` and `native/target`, `cargo install --locked cargo-deny@0.20.2` (cached), the gates (the documentation gate included), a release build, then `pytest -m native tests/native/` (the contract test, section 5 N0). The `pre-commit` job sets `SKIP=cargo-fmt,cargo-clippy,cargo-test,cargo-deny` so no check runs twice | https://embarkstudios.github.io/cargo-deny/ |
| Dependabot | a `cargo` entry for `/native` in `.github/dependabot.yml` | |
| Ratchets | `scripts/codegen/check_file_size.py` covers `native/**/*.rs` (split by concern, as for Python); `ATTRIBUTION.md` lists every vendored or derived source (libjpeg-turbo, Little CMS, Triton protos, the nvJPEG bindings, Clipper) | |

## 4. Decisions a measurement settles

| # | Decision | State | Evidence or the measurement that settles it |
|---|---|---|---|
| 4.1 | Model calls through the model server over shared memory, or an in-process runtime | settled for N0 to N3: the model server | pipeline: per-photo calls over shared memory reach the models' ceiling (COCO 197 against Z 192 to 197) at 0.5 ms of host CPU per photo; an in-process runtime has no measured gain to offer, and process separation is what meets the fault rule. Reopened only by a named future measurement that shows the model server as the bound (its queue time or its own CPU per call at the card ceiling, on two cards). Bar for the other filling then: plans F and C faster by more than the spread at equal parity (G4), host CPU and device memory not above, the kill and restart rows passed with start time inside G7, every model's second build path through plan 13's precision gates, and the operational cost written down. The loser is deleted |
| 4.2 | Async front or threads | settled in shape, open in size | pipeline: OS threads for decode plus a small tokio runtime for gRPC was the simplest fit (0 errors at 256 calls in flight). Open: the front's own CPU per photo for 10 MB request bodies, measured in N0 by thread role (cgroup per-thread time); rule: under 2 ms per photo on 20 MP at 64 in flight and never the named bound, else the response build moves to the finish pool |
| 4.3 | One process or several per card | settled: one | pipeline: one native process beat four Python processes on input (20 MP 226 to 255 against 222 to 244; COCO 10,155 against 3,398) |
| 4.4 | Decode threads against cores | settled default, swept around it | pipeline: knee at 24 threads = the physical cores (255.5), worse at 32 (222) and 48 (209). Default = physical cores available (2.1); N0 sweeps 0.5x, 0.75x, 1x of it and the NUMA arm; lane device memory recorded (it bounds the count on small cards: section 2.3) |
| 4.5 | Deep batching across photos for bulk ingest | settled: none | deep batches gave larger model batches at 3 to 5x the latency and no more photos/s at 20 MP; the per-photo Python cost they hid is gone in a native engine; the pipeline reached the ceiling without them |
| 4.6 | An engine-side batcher across photos | settled: not built | pipeline: the server's batcher on resident inputs reached the ceiling (server batches 20 to 30 at 128 in flight) |
| 4.7 | Where the general parts live | settled: `native/gpu_frame` in this repository | the owner's server as measured lacks the encoder, decode once for many outputs, crops, faces, text and renditions (section 3) |
| 4.8 | Which nvJPEG bindings | settled: the server's, nvJPEG part (section 3) | |
| 4.9 | Server receive windows: fixed 1 MiB / 2 MiB or adaptive | open | N0 arm on 20 MP at 64 in flight: front CPU per photo and p99 |
| 4.10 | Launch and finish thread counts | open | N0 rows by thread role; the counts then derive from cores per section 2.1 |

## 5. Phases

The shortest path to the switch: N0, then three N1 lanes in parallel, N2 overlapping the end of N1,
then N3. Nothing waits for plan 14 phase 3.

**Who does what (the multi-agent rule).** A lane writes code and runs light checks only: unit tests
(each seen failing first with a deliberately wrong constant), the Rust gates of 3.1, hooks on touched
files, the contract test (no GPU). Every GPU, compose or container run is the coordinator's hold on
the card the coordinator names (GPU 2 for N0; GPUs 0 and 2 for N2's two-card row), after checking
peer sessions' use of the card. A lane lists such runs as owed, never runs them.

**Evidence for the paper (N0 and the exit of every phase).** Every speed claim is a table that names
the set (the public ones: COCO, the Commons set, Open Images or Objects365; and set P, "a private set of
real-world 20 MP vehicle photos", with its count and megapixels, no source), the hardware (card, PCIe generation, CPU model and physical cores, other load on the card and
the host at start and end, the 1-minute load), the versions (engine commit, image digests,
toolchain, CUDA libraries, model server, the SHA-256 of every `.plan`), the exact command, the bound
that remains with its evidence, and the Python engine in the same hold. Raw files (JSON lines,
statistics snapshots, profiler output) stay under `artifacts_local/native_engine/<phase>/` and are
never bulk-deleted; the summary row goes into section 9.

### N0: the first slice of the product

**Goal.** The smallest native engine that serves protocol version 1 for the two plans the decisions
name: plan F (`detect`, `embed_frame`, crops `none`) and plan C, the ingest plan the API sends
(`ingest_engine_plan`: `detect`, `embed_frame`, `embed_crops` with `crop_embed` = model
`pe_crop_encoder`, size 224, batch 32, recipe `letterbox`, fill 128, and `crops: inline`), with GPU
decode, orientation, blur sums and model calls over CUDA shared memory, measured against the Python
engine in the same holds. Product code in the file layout below: N1 adds branches and routes, not a
rewrite.

**Entry.** This revision approved; a worktree and branch for the lane from the integration head.

**Scope in.** The front (every KServe call of 1.1, statuses and their order, the `outputs` filter,
the oversize mapping); the plan parser for the whole vocabulary (an unbuilt branch answers
`unsupported_branch` as the protocol defines); admission, route and budget, in-work bound, region
window and slab; region create, prefix purge, register, re-register (section 7, "region unknown"),
unregister then free; the `ModelCalls` filling with deadlines and cancellation; nvJPEG lanes;
orientation (`orient.cu`, a pure permutation, new); the kernels `resample_h`, `resample_v`,
`lap_sums`; host weights; `detect` (letterbox, call, settle with `undo_letterbox` and `crop_rect`,
blur sums of the frame and every detection); `embed_frame` (centre square); `embed_crops` with
`embed_classes` (letterbox squares on the fill, antialiased when the box is larger than the square,
plain bilinear when smaller, chunks of `batch`); `crops: inline` (finish pool, libjpeg-turbo 3.1.2,
`OP_FRAME_ENGINE_JPEG_QUALITY`); readiness, health tick, context-lost handling, SIGTERM drain;
settings and metrics with the Python names; the stub pipeline for the contract test.

**Scope out (N1).** Stored `boxes` and `if_detected` rounds, the host route (a photo that needs it
answers `decoder` in N0), the colour step (the N0 sets carry no converting profile), the `body`
transport, `OP_FRAME_ENGINE_CROP_MAX_SIDE` above 0 (`cv2.INTER_AREA`; N0 refuses a non-zero value at
start, the default is 0), faces, OCR, renditions.

**Inputs.**

| Input | Value |
|---|---|
| Python reference | the frozen engine at N0's second commit (the kernel move), whose parent carries lane C1; image built from `docker/frame-engine/Dockerfile` there; every row names that commit and the image digest. Arms: committed defaults (workers 16, lanes 16, encode threads 8, switch interval 0.1 ms, region 1,024 MiB, budget 2,048, pool 4,096), and one arm with lanes and workers at the physical cores (review M12); the better Python row is the reference |
| COCO set | pin `scripts/datasets/manifests/coco_bench_2000.json`, fetched with `python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_bench_2000 --bench-set 2000 --seed 20261009 --manifest scripts/datasets/manifests/coco_bench_2000.json` (`--verify-only` on later runs) |
| 20 MP set | pin `scripts/datasets/manifests/commons_hires_500.json`, fetched with `python scripts/datasets/fetch_hires_set.py --out data/samples/hires_500 --n 500 --seed 20261010 --manifest scripts/datasets/manifests/commons_hires_500.json` |
| Lists | new `scripts/bench/frame_engine_sets.py`, header reads only, the rule of the input-pipeline measurement: keep baseline or extended-sequential 8-bit 3-component JPEGs whose embedded profile, if any, is sRGB-class by `src/utils/icc_srgb.py`; COCO: up to 2 MP, `random.Random(1)` shuffle in pin row order, first 1,000; 20 MP: 18 to 24.5 MP, seed 3, 120; writes each list, its SHA-256 and the excluded counts by reason. Set P: lists of 200 and 2,000, same rule. The comparison's main rows (H1 to H3) run on the standard benchmark set's lists: public COCO lists of 500 and 2,000 photos and set P lists of 200 and 2,000; the 120-photo public large-photo list is a secondary, reproducible row. The lists are not the pipeline's byte for byte, so no pipeline number is a reference: the Python engine is measured again in the same hold |
| Model server | `openprocessor-triton` built from `Dockerfile.triton` at the N0 base commit (`FROM nvcr.io/nvidia/tritonserver:26.10-py3@sha256:b404f7de5dbeeefe0b17d7d149745a50d0989a25d0d831433b5ba63c716c2fcd`, TensorRT 11.4.0.106), tagged as the compose file names it at that commit; the flags of `docker-compose.yml`'s `triton-server` command plus a second `--model-repository=/stand_in` and the loads below |
| Models | `yolov11_small_trt_end2end` (`make export-models`), `pe_image_encoder` and `pe_crop_encoder` (`make export-pe`), configs as committed; the SHA-256 of each `.plan` recorded |
| Stand-in models | below; generated by new `scripts/bench/frame_engine_stand_ins.py` into an untracked directory mounted at `/stand_in` |
| Toolchain and bases | Rust 1.99.1; cargo-deny 0.20.2; builder `rust:1.99.1-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7`; runtime `debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2` (both read from Docker Hub on 2026-11-15; re-read only by a deliberate pin bump); NVIDIA wheels unpacked in a stage on `python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90` (the Python engine's pinned base) |
| Results | `artifacts_local/native_engine/n0/` |

**Stand-in models (the decisions' identity models).** Two kinds, loaded beside the real models;
both engines reach them through their normal transport, because the plan names the models.
Returning the input as an output would not help: an engine asks the model server only for the
outputs it reads, so the frozen Python engine would never see the extra tensor.

- Null models, for engine capacity (G2): the real model's config with another name and a TensorRT
  plan from a tiny ONNX graph (cast one input pixel, add zero times it to a constant), built by the
  script with `trtexec` in the `model-export` image at the real model's profile. They cost the card
  nearly nothing and answer a constant: the detector five fixed rows (centred boxes of 16, 48, 160,
  320 and 600 px in the 640 canvas, score 0.9, classes 0, 2, 5, 7, 16: crops both smaller and larger
  than the 224 square), the encoders a fixed unit vector. Their own alone row Z0 is measured in
  every hold.
- Capture models, for the kernel-bytes comparison (G4, model inputs): Python-backend models (the
  backend is in the model server image; `ocr_pipeline` uses it) with the same names and outputs as
  the null models plus `FORCE_CPU_ONLY_INPUT_TENSORS` `yes` and `CAPTURE_DIR`; `execute` writes each
  input row to `<CAPTURE_DIR>/<model>/<sha256>.bin` and appends its order, digest and shape to
  `index.txt`. The fixed boxes make both engines cut the same crops from every photo.

```text
name: "null_detector_640"
platform: "tensorrt_plan"
max_batch_size: 64
input [ { name: "images" data_type: TYPE_UINT8 dims: [ 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 300, 6 ] } ]
dynamic_batching { preferred_batch_size: [ 8, 16, 32, 64 ] max_queue_delay_microseconds: 25000 }
instance_group [ { count: 2 kind: KIND_GPU gpus: [ 0 ] } ]
```

`null_encoder_336` and `null_encoder_224` are `pe_image_encoder`'s and `pe_crop_encoder`'s configs
with the name changed (`images` UINT8 [3, S, S] to `image_embeddings` FP32 [1024], max batch 32,
preferred 4, 8, 16, 32, delay 15 ms). The capture models are `capture_detector_640`,
`capture_encoder_336`, `capture_encoder_224` with `backend: "python"`.

**What N0 keeps of the pipeline's code** (the throwaway tree; N0 is written to this layout and its
tests, not grown from that tree).

| Pipeline file | N0 |
|---|---|
| `geom.rs` | kept: `letterbox_geometry`, `whole_square_geometry`, numpy's pairwise sum, the weight blob packing, into `native/frame_engine/src/geometry.rs`; the non-antialiased branch of `filter_weights` added for small crops; its weight check script becomes the golden generator |
| `gpu.rs` | kept: the NVRTC sequence (options, `nvrtcGetCUBIN`, `cuModuleLoadData`) into `gpu_frame/src/kernels.rs`; the launch arithmetic of `resample_into` into `frame_engine/src/kernels.rs`; the weight upload on the lane's stream with a wait before publishing (the fix of its illegal-address crash: a synchronous copy from pageable memory may return before the DMA lands). Rewritten: the arch from the device, errors instead of panics, a byte-capped cache, buffers from the frame pool instead of grow-only buffers |
| `triton.rs` | kept: region create (`malloc_sync`, `cuIpcGetMemHandle`, register by raw handle), unregister then free, the shared-memory input builder, the channel settings, into `gpu_frame/src/model_server.rs`. Added: prefix purge, re-register, the slab, the trait, per-request deadlines |
| `nvjpeg.rs` | replaced by the bindings of section 3; its call order is the cross-check |
| `main.rs` | not kept as engine code; its closed loop, NVML sampler (50 ms) and statistics deltas are the start of the loadgen |
| `build.rs` | rewritten: kernels by `include_str!` from `native/frame_engine/kernels/`, never extracted from Python; `protox` compile kept; no link line (nvJPEG is loaded at run time) |
| `Cargo.toml`, `Cargo.lock` | the pins and features kept (cudarc, tonic, tonic-prost, prost, protox, tokio, crossbeam-channel); added: libloading 0.9.1, turbojpeg 1.5.1, turbojpeg-sys 1.2.0, memchr, prometheus-client 0.26.0; nvml-wrapper in the loadgen only |
| `proto/` | kept: Triton r26.10 `grpc_service.proto`, `model_config.proto`, with the tag in a header line |

**Commits, in order.**

1. The Rust workspace and its gates: `native/Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`,
   `deny.toml`, `clippy.toml`, empty crates `gpu_frame` and `frame_engine` with one test each, the
   hooks and the CI job of 3.1, the Dependabot entry, the ratchet scope, `CHANGELOG.md`, and the
   offline command in `CLAUDE.md` and `CONTRIBUTING.md` gaining `and not native` with a line for
   `-m native`.
2. The kernel move: `native/frame_engine/kernels/prep.cu`, `colour.cu`, `faces_warp.cu`,
   `ocr_warp.cu`, byte-identical to today's `_SRC` strings; `frame_engine/prep_kernels.py`,
   `colour.py`, `faces/kernels.py`, `ocr/kernels.py` read the files; `docker/frame-engine/Dockerfile`
   copies the directory. The only change to the frozen engine (Q5). Owed to the wave: the GPU tier
   (`test_gpu_decode_parity.py`, `test_gpu_renditions.py`, `test_gpu_colour.py`,
   `test_gpu_crop_encoder.py`) passes unchanged.
3. The contracts: the generator, both JSON files, the hook pattern. The wire module first rejects
   four plan quirks the API never sends (review m4): `version` given as `true` or `1.0` (Python
   equality accepts both), `NaN` and `Infinity` literals, duplicate keys (`bad_json`), with cases;
   then the native parser equals Python's on every case.
4. Onward, one concern per commit: `gpu_frame` (lanes, frame pool, region and slab, admission,
   kernel loader, model calls), then the engine (settings, front, wire, admission wiring, geometry,
   routing, photo, branches, metrics, health), the stub pipeline and the contract test, the
   loadgen, the bench scripts, the image and the test bed.

**Files.**

| Path | What |
|---|---|
| `native/gpu_frame/src/lib.rs`, `cuda.rs`, `nvjpeg.rs`, `nvjpeg/ffi.rs`, `kernels.rs`, `region.rs`, `slab.rs`, `budget.rs`, `frame_pool.rs`, `model_calls.rs`, `model_server.rs`, `proto/` | the general crate (2.0); every `unsafe` call in `cuda.rs`, `nvjpeg/ffi.rs` and `region.rs` only, each with its SAFETY note |
| `native/frame_engine/src/main.rs`, `settings.rs`, `front.rs`, `wire.rs`, `admission.rs`, `geometry.rs`, `jpeg_header.rs`, `photo.rs`, `kernels.rs`, `branches/detect.rs`, `branches/embed.rs`, `branches/crops.rs`, `metrics.rs`, `health.rs`, `stub.rs` | one concern each, named after the Python module it replaces; `stub.rs` is the model-free pipeline |
| `native/frame_engine/kernels/` | the moved sources; `orient.cu` new |
| `native/frame_engine/tests/golden/` | weights, geometry and routing cases generated by `scripts/codegen/frame_engine_goldens.py` from the Python functions (data committed; the script is deleted at N3) |
| `native/frame_engine_loadgen/` | `op-fe-loadgen`, as built: `engine` mode (closed loop over a list held in memory, protocol version 1 to any engine: either engine is measured by the same client; the plan text comes from a file the parity tool writes with the API's builder, `--plan-file`, so the plan rule has one copy), `z` mode (models alone through `gpu_frame`'s `ModelCalls`, inputs resident; for plan C it replays each photo's crop count recorded by a Python parity run, refused for another list), `ready --model` and `regions --prefix` (the gates of the holds); one JSON line per row (`op-fe-loadgen/1`): photos/s, p50, p99, failures by key, host CPU of the measured process from its cgroup `cpu.stat`, card busy and per-process device memory from NVML every 50 ms, model-server statistics deltas (mean batch, queue time per request), and the evidence fields of section 5. Layout differs from the first wording in one respect: `--plan frame|ingest --models real|...` moved to the parity tool's `plan` command, which writes the file the client takes |
| `scripts/bench/gpu_prep_parity.py` | extended (no new parity tool): `--plan frame|ingest`, `--models real|null|capture`, `plan` (writes the plan text), the rows of 1.2 as a verdict with the Python-against-Python floor (`compare --floor-a --floor-b --in-flight`), `capture --crop-counts-out`, and `compare-inputs DIR_A DIR_B` (per model, the sorted digests of every captured input row must be equal; differing rows are matched by arrival order and diffed byte by byte) |
| `scripts/bench/frame_engine_sets.py`, `frame_engine_stand_ins.py`, `frame_engine_capture_model.py`, `frame_engine_hold.sh`, `frame_engine_faults.sh`, `frame_engine_bed.sh` | the lists; the stand-ins and the Python-backend model they copy (`model.py` of each capture model, which writes per arm into `/capture/<arm>/<model>/`); one hold (every arm, alternating, `flock -o` on the coordinator's hold file, `timeout --kill-after` innermost, the load gate, the device guard); the kill loop and the model-server restart (`docker kill -s KILL` by exact name at 32 in flight, checks of the model server after each); the shell parts both scripts share (lock, gates, bounded command, options) |
| `docker/frame-engine-native/Dockerfile` | stages: wheels (the pinned Python base, `pip download` of the three NVIDIA wheels at the versions of `docker/frame-engine/requirements.txt`, read from it, libraries copied out); build (the Rust base, libjpeg-turbo 3.1.2 with cmake and nasm checked by SHA-256, `cargo build --release --locked` in a BuildKit cache mount, `RUSTUP_TOOLCHAIN` set to the pinned version); runtime (the Debian base, non-root user `engine` in group `video`, `LD_LIBRARY_PATH` to the copied libraries, `op-frame-engine health` as the health check). The wire contract reaches the build as the named context `contracts_json`: the root `.dockerignore` keeps `contracts/` out of every context |
| `docker/frame-engine-native/compose.n0.yml` | the test bed, project `op-n0`: services `triton-server` (container `op-n0-model-server`, GPU by `N0_GPU_ID`, default 2; gRPC on 4620; `--model-repository` twice; `--allow-client-shm=true`), `engine-native` (4621, metrics 4623) and `engine-python` (4622, metrics 4624), each under a fixed container name, `init: true`, `memlock` unlimited, distinct `OP_FRAME_ENGINE_INSTANCE` names (an engine unregisters every region of its own prefix at start) |
| `tests/native/test_native_front_contract.py` | marker `native`: the release binary with `--pipeline stub` (no CUDA) driven by `tritonclient` and `src/clients/frame_engine.py`: every case of today's `test_front.py` (bytes round trip, each branch combination's tensors, stored boxes as FP64 rows, bad request, unknown model, readiness and draining, metadata, failures to statuses, in-flight cap, deadline with its stage), the retry and probe cases of `tests/clients/test_frame_engine_retry.py` and `test_frame_engine_probe.py`, an over-limit message answering `RESOURCE_EXHAUSTED`, the `frame engine: ` prefix through grpc-message encoding, trailers-only errors, 10 MB bodies, an idle channel pinged every 30 s for 65 s without GOAWAY |
| `docker/frame-engine-native/Dockerfile` | stages: wheels (the pinned Python base, `pip download` of the three NVIDIA wheels by version, libraries copied out); build (the Rust base, libjpeg-turbo 3.1.2 with cmake and nasm, `cargo build --release --locked`, a cached dependency layer); runtime (the Debian base, non-root user `engine` in group `video`, `LD_LIBRARY_PATH` to the copied libraries, `op-frame-engine health` as the health check) |
| `docker/frame-engine-native/compose.n0.yml` | the test bed, project `-p op-n0`: model server on GPU 2 (gRPC published on 4620), native engine (4621, metrics 4623), Python engine (4622, metrics 4624), `init: true` on both engines, `memlock` unlimited, `--allow-client-shm=true` |

**Unit tests in the lane (no GPU).** Wire parsing against the plan cases; admission and budget
fairness; the in-work bound; slab first fit, coalescing and the window rule; deadline arithmetic;
routing (end-of-image before reservation) and geometry against the goldens, weights bit for bit on
every tuple of the three lists; the BYTES packing; the status map of every error class including the
oversize mapping; the sticky-code table; settings ranges and the refused settings; panic isolation
on a pool thread; cancellation at each hand-off.

**The holds (coordinator, GPU 2, exclusive, bounded, writing files).** Every row: failed 0, the
1-minute load at start and end (gate under 16), the model server restarted only between holds.

| Hold | Rows |
|---|---|
| S0 smoke, as soon as `detect` runs (90 s) | region registered; one plan-F request on 3 COCO photos; the capture model's canvas digests equal the Python engine's; idle 65 s then a request with no GOAWAY in the model server's log. A failure here stops N0 for a fix before anything else is built |
| H1 parity | `gpu_prep_parity.py` at 1 and 32 in flight, plans F and C, 200 photos of the COCO 500 list, 100 of the 120-photo public large-photo list and the set P 200 list; Python against itself twice for the floors; `compare-inputs` on 50 frames per set with the capture models |
| H2 rate and ceiling | plans F and C at 1, 8, 32, 64, 128 in flight on COCO, 20 MP, set P; both engines, two runs per arm, alternating; Z rows for F and C at 32 and 64; the native input-only row for d (G1) |
| H3 capacity | null models, plans F and C, COCO and 20 MP, 32, 64, 128 in flight, both engines; Z0 rows |
| H4 sweeps | lanes at 0.5x, 0.75x, 1x of the physical cores and the NUMA arm (4.4); server windows fixed or adaptive (4.9); thread counts by role (4.2, 4.10) |
| H5 memory, size, start | device memory idle and peak per process, host RSS peak, the accounting against NVML; `docker image ls`; container start to `ModelReady` and process start to ready, both engines |
| H6 faults | 10 kills at 32 in flight; a model-server restart with the engine up |

Commands (from the lane's worktree; ports as in `compose.n0.yml`; the developer guide page
"Running the native engine comparison (N0)" has them with every argument):

```text
python scripts/bench/frame_engine_sets.py --preset coco --manifest scripts/datasets/manifests/coco_bench_2000.json \
    --images-dir data/samples/coco_bench_2000/images --out-dir artifacts_local/native_engine/n0/lists   # and --preset hires
python scripts/bench/frame_engine_stand_ins.py generate --onnx --out artifacts_local/native_engine/n0/stand_in
python scripts/bench/frame_engine_stand_ins.py plan-commands --run
docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml build
docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml up -d
timeout 600 scripts/bench/frame_engine_hold.sh --hold-file <given by the coordinator> --gpu-index 2 \
    --gpu-bus-id <the card's bus id> --lists artifacts_local/native_engine/n0/lists \
    --capture-dir artifacts_local/native_engine/n0/capture --rows s0 --out artifacts_local/native_engine/n0/s0.jsonl   # then h1, h2, h3
scripts/bench/frame_engine_faults.sh --mode kill --hold-file <...> --gpu-index 2 --gpu-bus-id <...> \
    --list <list> --plan-file <plan> --out artifacts_local/native_engine/n0/h6.jsonl                    # and --mode server-restart
python scripts/bench/gpu_prep_parity.py plan --plan ingest --models real --out <plan file>
timeout 120 native/target/release/op-fe-loadgen engine --url 127.0.0.1:4621 --list <list> \
    --plan-file <plan file> --in-flight 32 --warmup-s 5 --window-s 20 --stats-url 127.0.0.1:4620 \
    --gpu-bus-id <bus id> --out <file>
timeout 900 python scripts/bench/gpu_prep_parity.py capture --url 127.0.0.1:4621 --images <list> \
    --plan ingest --models capture --concurrency 1 --out <file>
docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml down
```

**Go when every bar holds:** G1 and G2 on plans F and C; G3; G4 for `detect`, `embed_frame`,
`embed_crops` and `crops`; G5, G6, G7; G8 (N0 part); G9. **No-go** when a bar misses with its bound
inside the native engine and no fix inside N0, when model inputs differ for a reason not found in the
loader, the options or the geometry port, when the region cannot be registered or read, or when a
kill leaves the model server restarted, not ready or above its baseline. A miss is reported with its
number and its bound; a bar moves only in writing (plan 14 section 6).

**Thrown away after N0.** `compose.n0.yml`, the `decoder` answer for host-route photos, and the
refusal of a non-zero crop side cap (both replaced in N1a). Everything else is product code.

### N1: every branch, each with its parity gate (three lanes in parallel)

**Entry.** N0 a go and merged. **Exit.** Every branch any caller sends and every branch the Python
engine builds passes its rows of 1.2 in the wave sweep; the branch rows recorded under the evidence
clause.

| Lane | Adds | Gate (1.2) |
|---|---|---|
| N1a, ingest rest | stored `boxes` and `if_detected` with the late rounds; the host route with the accepted-format corpus and strict JPEG decode; the colour step (classify, table single-flighted per profile, gather); the `body` transport; the crop side cap with OpenCV's area filter ported; every refusal status | the four-way ingest rows through the API (plan 13's E1 script): crop counts, pixel sizes and decoded pixels, crop vectors, blur sums; the 88-profile corpus; the format corpus and a damaged-file corpus (truncated, bad Huffman, extra bytes after the end marker) with the same status as Python on every file |
| N1b, faces | `faces` (whole frame, coarse pass, tiles), `face_embed` (warp kernel, chunks of `batch`), anchors and NMS on the finish pool | `tests/gpu/test_gpu_faces.py`; LFW, CALFW, CPLFW and WIDER FACE as 1.2 says; the numpy reference functions the test imports from `frame_engine.faces` move into `tests/gpu/` |
| N1c, text and renditions | `ocr` (detector input, DB boxes, line warp, batches by width, CTC), `derivatives` (Lanczos weights, nvJPEG encode); the recogniser dictionary `models/paddleocr_rec_trt/ppocrv6_dict.txt` copied into the image | `tests/gpu/test_gpu_ocr.py` and the COCO-Text rows; `test_gpu_renditions.py`, `test_gpu_colour.py` |

### N2: packaging, profiles, faults, docs and the sweep

**Entry.** N1a merged (N2 starts while N1b and N1c finish). **Exit.** All rows below green in one
sweep on the merged tree; speed rows recorded under the evidence clause.

- Image: Trivy scan clean of high and critical findings (the vendored C libraries are in its scope);
  non-root; G6 and G7.
- Settings: every `OP_FRAME_ENGINE_*` with the Python ranges (1.1); `env.template` and the sizing
  page say the new `OP_FRAME_ENGINE_LANES` default.
- Profiles (E4): sizes per profile from measured memory, the 12 GB card starting inside its memory
  and running plan A on the mix with failed 0, the accounting asserted against NVML there; the
  profile files, the installer mirror, its test and the sizing page changed together.
- Faults: 50 kills at 32 in flight; a model-server restart (engine ready within 15 s of the server's
  gRPC port); the server recreated on a new address (first good call within about 1 s); SIGTERM
  drain; a sticky error injected (a test build with an out-of-range kernel write) ends in exit 1 and
  a clean restart; a region unknown to a restarted server re-registered.
- Metrics: names and labels of `/metrics` equal to the Python engine's, `process_*` included (a
  scripted diff).
- Two cards: both A6000s (GPUs 0 and 2), plans F and C on COCO and the 20 MP set, two arms: one
  model server spanning both cards with one engine, and one engine plus one model server per card;
  decode threads pinned to each card's socket as an arm; host cores busy recorded (section 0a
  predicts the host bound at 20 MP).
- Docs: `docker/test/README.md` GPU tier, the docs-site engine and sizing pages (concise user page,
  detailed developer page), `env.template`, `CHANGELOG.md`.
- The sweep: full offline suite, `pytest -m native`, the GPU tier on the native engine, the four-way
  ingest rows, the 600 s soak of configuration D with failed 0, every number with host CPU per photo
  per process (plan 14 section 4).

### N3: the switch and the deletion

**Entry.** N2's sweep green; G1 to G9 met on plans F and C on the merged tree, and plan A measured
(G1 with Z_A, the replay of the Python engine's recorded calls for plan A). **Exit.** One engine in
the tree and in the stack, a green sweep, a ledger entry, the final table of plans F, C and A
against the last Python measurement under the evidence clause (the paper's main table), and the general GPU crate (`gpu_frame`) moved into the owner's separate open-source Rust serving project when the owner publishes that project (owner decision 2026-11-15; until then it stays in this repository, with no engine, plan or wire type in its interface).

In one change: the compose service keeps its name and image name (`openprocessor-frame-engine`) and
points at the native build; `docker/frame-engine-native/` becomes `docker/frame-engine/`; the guard
test `tests/test_no_python_frame_engine.py` runs `git grep` for the Python engine's paths, imports,
test directory, requirements file, `python -m frame_engine` and the retired setting, and must find
nothing outside `CHANGELOG.md` and `docs/design/` (plans are records; their links stay checked by
the docs checker). The table below is every tracked file that `git grep -l -E
"frame_engine|frame-engine|FRAME_ENGINE"` finds outside `frame_engine/` and `tests/frame_engine_tests/`
at `7fb83e660` (146 files), by action:

| Action at N3 | Paths |
|---|---|
| Deleted | `frame_engine/` (all), `tests/frame_engine_tests/` (all; their cases live on as Rust tests, goldens and `tests/native/`), `docker/frame-engine/requirements.txt`, the Python `docker/frame-engine/Dockerfile` (replaced by the moved native one), `scripts/codegen/frame_engine_goldens.py`, `scripts/codegen/frame_engine_face_goldens.py`, `scripts/research/gpu_pipeline/harness_common.py`, `pr0_stage_split.py`, `shipped_pipeline.py`, `README.md` (research harness importing the shipped modules by path) |
| Moved (`git mv`, in the commit that deletes `frame_engine/`) | `frame_engine/faces/scrfd.py` and `frame_engine/faces/alignment.py` into `tests/gpu/`: the numpy reference functions `tests/gpu/test_gpu_faces.py` imports. N1's row named this move for N1b; it waits for N3 because the Python engine imports both files and its image copies only `frame_engine/` (coordinator, 2026-11-15) |
| Imports or runs the Python engine: rewritten | `tests/clients/test_frame_engine_retry.py`, `tests/clients/test_frame_engine_probe.py` (onto the native front with `--pipeline stub`), `docker/test/fake_frame_engine.py` (a small KServe front of its own in `docker/test/`), `docker/test/compose.yml`, `docker/test/compose.built.yml`, `docker/test/README.md`, `tests/compose/test_fake_frame_engine.py`, `tests/curation/test_crop_is_stored_box.py` (geometry from `src/domain/image_geometry.py`), `tests/test_frame_engine_profile_sizes.py` (ranges from the generated contract), `tests/test_decode_route_vocabulary.py` (scans `native/`), `tests/test_crop_rule_single_copy.py` (names the native crop code), `tests/gpu/test_gpu_faces.py`, `tests/gpu/test_gpu_renditions.py`, `tests/test_icc_srgb.py`, `tests/test_ocr_uint8_configs.py` |
| Names the directory in a gate: scope moved to `native/` | `.pre-commit-config.yaml` (seven `files:` patterns), `pyproject.toml` (six entries), `scripts/codegen/check_file_size.py`, `check_vulture.py`, `check_file_locks.py`, `check_duplication.py`, `check_ruff_ratchet.py`, `check_layering.py` (rule 8 and `_ENGINE_STAND_IN`), `tests/codegen/test_check_layering.py`, `scripts/docs/check_docs_vs_code.py` and `tests/test_env_surface.py` (both must read `OP_FRAME_ENGINE_*` from `native/**/*.rs`, or every engine setting reads as unused) |
| Build, release, CI | `.github/dependabot.yml` (the pip entry goes), `.github/workflows/security-scan.yml`, `scripts/lib/image_keys.sh`, `scripts/release/build_and_publish.sh` (key `frame_engine` kept, new Dockerfile), `tests/test_release_build_and_publish.py`, `tests/compose/test_frame_engine_service.py`, `tests/compose/test_images_build_and_overlays.py`, `docker-compose.yml` (health check `op-frame-engine health`; comments naming Python modules), `docker-compose.dev.yml` (the `./frame_engine` and `./src/domain` source mounts go), `docker-compose.ingest-focus.yml`, `docker-compose.faces-off.yml`, `export/sync_model_seed.py`, `export/README.md` |
| Settings | `env.template`, `src/config/retired_env.py` (adds `OP_FRAME_ENGINE_SWITCH_INTERVAL_MS`), `scripts/lib/retired_env.sh` |
| Docs | `CLAUDE.md`, `ATTRIBUTION.md`, `CHANGELOG.md` (entry), `docs/ARCHITECTURE.md`, `docs/OCR.md`, `docs/Technical/TRITON_BEST_PRACTICES.md`, `docs/design/sp_gpu_prep_concurrency.md` (its link to `frame_engine/engine.py` would break), `docs/releases/RELEASE_CHECKLIST.md`, `docs-site/docs/architecture/frame-engine-and-ingest.mdx`, `deployment/gpu-sizing.mdx`, `operations/frame-engine.mdx`, `operations/ingest-stage-timing.mdx`; the plans `docs/design/v070_rebuild/13_backend_speed_plan.md`, `13a_ingest_on_engine_plan.md` and `17_native_engine_plan.md` closed |
| Docstrings naming `frame_engine/` paths: text updated | `src/clients/frame_engine.py`, `src/config/curation.py`, `src/domain/decode_route.py`, `src/domain/image_geometry.py`, `src/domain/jpeg_markers.py`, `src/services/curation/ingest_gpu_decode.py`, `src/utils/pe_preprocess.py`, `scripts/bench/gpu_prep_parity.py`, `tests/curation/engine_answer.py` |
| Service name, model name, client or kept setting only: checked, unchanged | `config_templates/profiles/full.json`, `minimal.json`, `small.json`, `standard.json`; `INSTALLATION.md`; `openprocessor`; `setup-openprocessor.sh`; `monitoring/prometheus.yml`; `docs/README.md`; `docs-site/sidebars.ts`; `docs-site/docs/architecture/overview.mdx`, `configuration/advanced.mdx`, `configuration/basic.mdx`, `getting-started/compose-profiles.mdx`, `operations/ingest-load-profile.mdx`, `operations/monitoring.mdx`, `operations/release-acceptance.mdx`, `operations/upgrade-and-backup.mdx`; `scripts/bench/baseline_report.py`, `detector_capture.py`, `ingest_profile.py`, `ingest_profile_report.py`, `run_baseline.py`; `scripts/lib/cli_gpu.sh`, `config.sh`, `gpu.sh`; `scripts/release/acceptance_phases.sh`, `acceptance_run.sh`; `scripts/research/near_dup/live_ingest.py`; `src/app/lifespan.py`, `src/app/resources.py`; `src/config/faces.py`; `src/domain/processing_plan.py`; `src/services/labeling/vlm_url_policy.py`; `src/services/upgrade/from_060/steps/serving_profile.py`; `src/services/curation/ingest.py`, `ingest_service_factory.py`, `reprocess_gpu_embed.py`, `dataset_import/job.py`; `src/tasks/jobs_a_support.py`; `tests/bench/test_detector_capture.py`, `test_ingest_profile.py`, `test_ingest_profile_report.py`, `test_run_baseline_reps.py`; `tests/clients/test_frame_engine_client.py`, `test_frame_engine_failure_observability.py`; `tests/compose/test_ingest_focus_overlay.py`; `tests/curation/cross_project_leak/_env.py`, `engine_fakes.py`, `test_ingest_gpu_decode.py`, `test_ingest_profile_consumers.py`, `test_ingest_profiles.py`, `test_reprocess_gpu_embed.py`; `tests/gpu/test_gpu_colour.py`, `test_gpu_crop_encoder.py`, `test_gpu_decode_parity.py`, `test_gpu_ocr.py`; `tests/installer/shims/docker`, `test_cli_status_lists_new_services.py`, `test_gpu_plan.py`, `test_image_tag_mode.py`, `test_ingest_gpu_decode_profile.py`, `test_lifecycle.py`, `test_model_setup.py`, `test_setup_release_pinning.py`; `tests/integration/ingest_fakes.py`; `tests/live/_engine.py`, `_ingest_support.py`, `test_live_engine_unavailable.py`, `test_live_ingest.py`; `tests/release/test_acceptance_review_phases.py`; `tests/test_frame_engine_wire.py`; `tests/test_retired_ingest_size_gate.py`; `src/domain/frame_engine_wire.py` |

Files added between `7fb83e660` and N3 that match are added to the table by the N3 lane from the
same command. No compatibility path, no second engine kept.

## 6. Interim lever: several Python engine processes

Not done: no further speed work goes into the Python engine. The measurement stays as the reason the
native engine spreads decode over all cores in one process: the engine's input code in four processes
did 3,398 photos/s on small photos against 650 in one, and 1.2 to 1.4x at 12 to 20 MP.

## 7. Risks and their early signals

| Risk | Early signal | Answer |
|---|---|---|
| `cudarc` is pre-1.0 | a minor bump breaks the build | exact pins; raw calls only in `gpu_frame`'s `cuda.rs`, `nvjpeg/ffi.rs`, `region.rs` |
| Region unknown to the model server | a region-using call refused after a server restart | the shared-memory extension specifies no error text and no lifetime across restarts (https://github.com/triton-inference-server/server/blob/r26.10/docs/protocol/extension_shared_memory.md): re-register on `CudaSharedMemoryStatus` in the health tick and on any `INVALID_ARGUMENT` or `NOT_FOUND` class error from a call that uses the region; the message "Unable to find shared memory region" is a hint only; the lifetime rule is empirical (N2 restart row) |
| Kernel parity | the capture comparison differs | CuPy's exact options and the same NVRTC wheel; when options differ the lane reads CuPy's compiler module, never guesses |
| Lane memory on small cards | the 12 GB profile fails to start with the derived lane count | lane bytes are counted (2.3); on a small card the count derives from memory as well as cores (N2) |
| libjpeg-turbo builds | crop bytes or host-route frames differ | one version, 3.1.2, built from source for the native engine; Q6's rule where a setting cannot match |
| OCR post-processing | F1 below the Python branch's | the Clipper 6.4.2 behaviour, decided by the gate (section 3) |
| Build time | a clean image build over 10 minutes | cached dependency layer; one build per sweep |
| GPU test coverage | a GPU-tier test cannot run (faces and OCR need the uint8 builds) | listed as owed to the sweep, never skipped |
| Contract drift while N1 runs | the drift hook fails on the plan cases | the lane that changes the vocabulary regenerates the cases and the Rust parser in the same change |
| The stand-ins bound the capacity row | Z0 under 2x the native row | more stand-in instances; the row is not judged until Z0 clears it |

## 8. Questions for the coordinator (answered 2026-11-15)

| # | Question | Recommendation | Answer |
|---|---|---|---|
| Q3 | Outside contributions to `native/gpu_frame` and a later relicence | AGPL now (decided); accept outside contributions there only under a contributor agreement that allows a relicence, or record that a move needs their consent | Answer: AGPL now. Licence questions are deferred by the owner; until then the crate takes no outside contribution without the owner's yes |
| Q6 | Crop JPEG equality where a build or setting cannot match | decided: decoded pixels identical and sizes within 0.5%; bytes asserted where version and settings match (3.1.2 is planned to match) | As decided |
| Q7 | An over-limit request answers `RESOURCE_EXHAUSTED`, which the API retries until its 120 s deadline (today's behaviour, kept by decision) | keep for the switch; a separate API change could refuse it at once | Keep the status in N0 (equal answers). N1a adds a test that the API's upload limit is below the engine's message limit, so the API refuses an oversize photo before it is sent and the retry cannot happen |
| Q8 | The capacity bar G2 is applied on both plans F and C | confirm | Confirmed |

## 9. Results

None for the product yet; section 0.1 holds the pipeline's rows (prior evidence). N0 adds its rows
here in the form of the evidence clause.

## 10. Review trace

| Id | What changed | Where |
|---|---|---|
| B1 | vector rows pass on the Python-against-Python floor; exact rows limited to what the engine controls; tests named | 1.2 |
| B2 | capture models (Python backend) read model inputs through both engines' normal transport; `compare-inputs` | 1.2; N0 stand-ins, files |
| M1 | CuPy's three options, cubin through `nvrtcGetCUBIN`, arch from the device; no disk cache, start time judged by G7 | 3 (Kernel build); N0 keep table |
| M2 | weights: numpy's pairwise sum ported (pipeline: 1,281 sets byte-identical), tested on every tuple; renditions judged by pixels and PSNR | 1.2; N0 tests |
| M3 | N0 measures and gates plan C (crops inline, crop encoder chunks); the crop branch moved into N0; stored boxes stay in N1 | 0.2; N0 goal and scope |
| M4 | window rule `REGION_MIB / WORKERS`, round-2 staging inside it, oldest own call first | 2.2 |
| M5 | hard cap by the engine's own count, every allocation class and its payer, nvJPEG through a counting allocator, sync `cuMemAlloc` for the region, memory errors to `gpu_memory`, NVML assertion on 12 GB | 2.3; 1.4; N2 |
| M6 | channel settings both ways with doc citations; idle-ping case in the contract test and S0 | 2.4; N0 files and holds |
| M7 | accepted formats equal today's by corpus; strict decode via `TJPARAM_STOPONWARNING`; one libjpeg-turbo 3.1.2 | 3; 1.2; N1a |
| M8 | Clipper 6.4.2 behaviour (binding or port, decided by the gate); OCR gate on ground truth only | 3; 1.2; N1c |
| M9 | Rust gates, CI job, Dependabot, ratchets, attribution, contract hook pattern | 3.1; N0 commit 1 and 3 |
| M10 | deletion table from the search of the tracked tree, guard test | N3 |
| M11 | contract test: real API client against the native front with a stub pipeline, today's front, retry and probe cases | N0 files; 3.1 CI |
| M12 | one bar table derived from named rows; Python arm pinned by commit and digest with a cores arm; 20 MP CPU reported, not barred | 0.2; N0 inputs |
| M13 | colour port list adds `icc_curves.py`; lcms 2.19 as Pillow; no shared transform | 3 (Colour) |
| M14 | lane and coordinator split; S0 smoke first; commands | 5 (who does what); N0 holds |
| m1 | N0 inputs table: sets, lists, model server, models, bases, toolchain, results directory, loadgen command | N0 inputs, commands |
| m2 | no new parity tool: `gpu_prep_parity.py` extended; the loadgen measures either engine | N0 files |
| m3 | `ModelReady` per photo, check order, echoes, metadata, filter scope, validation rules, shapes in the contract | 1.1; 3 (Contract) |
| m4 | four plan quirks rejected in the wire module first, with cases | N0 commit 3 |
| m5 | sticky list from the documentation; 214 dropped | 1.3 |
| m6 | nvJPEG per-thread objects as documented; decoder sharing as practice; buffer lifetime with `save_stream = 0`; `bind_to_thread` | 2.0; 2.1 |
| m7 | re-register on the status call and error classes; the message is a hint | 7 |
| m8 | end-of-image check before reservation | 2.2 |
| m9 | cancellation flag at every hand-off | 2.2 |
| m10 | `catch_unwind` on pool threads, `panic = "unwind"`, a test | 1.3 |
| m11 | the server's bindings chosen (decision 4); `nalgebra` for alignment | 3 |
| m12 | `process_*` series kept; metrics HTTP server named | 3 (Metrics); N2 |
| m13 | `body` transport kept (N1a); switch interval refused by the native engine, retired at N3; crop side cap in N1a; dictionary file copied by N1c's image change | 1.1; N0 scope out; N1a |
| m14 | face and OCR bars use the Python branch's two-run difference; existing and new rows told apart | 1.2 |
| m15 | the desktop sentence removed; `CHANGELOG.md` in N0, N2, N3; the N0 `decoder` answer noted as thrown away | N0, N2, N3 |
| D1 | bits are the first gate; ground truth decides an explained difference | 1.2 |
| D2 | sticky set per the documentation, 214 not sticky, exit for restart unchanged | 1.3 |
| D3 | host formats: today's set, by corpus, same statuses | 3; N1a |
| D4 | `WORKERS` kept, `body` kept, switch interval retired at N3 | 1.1 |
| D5 | oversize maps to `RESOURCE_EXHAUSTED`; tonic's `UNIMPLEMENTED` stubs and h2 resets covered by the contract test | 1.1; N0 files |
| D6 | hard cap | 2.3 |
| D7 | vendored C libraries in Trivy's scope; a C++ toolchain only if the Clipper binding wins | N2; 3 |
| D8 | `native/` holds the two crates and the loadgen only | 2.0; N0 files |
| D9 | the coordinator runs holds on the card it names | 5 |
| D10 | relicence needs contributors' consent | 3; Q3 |
| Decisions Q1, Q2 | faces, text and renditions ported before the switch; C1 landed in the Python engine first (merged at `7fb83e660`) | N1; header |
| Decision Q3 | AGPL now, not a gate | 3; Q3 |
| Decision Q4 | two cards: N2 row; cores per card at its bound stated | 0a; N2 |
| Decision Q5 | kernel move, GPU tier unchanged in the same wave | N0 commit 2 |
| Decision Q6 | crop JPEG rule | 1.2; Q6 |
| Decision, added (N0 lane and holds) | product code, own worktree, light checks; GPU by the coordinator | 5 |
| Decision, added (gates in the first commit) | as asked | 3.1; N0 commit 1 |
| Owner direction 1 | model call behind one interface; 4.1 bar written; loser deleted | 1.5; 2.0; 4.1 |
| Owner direction 2 | module boundary and interface | 2.0 |
| Owner direction 3 | thread counts derived from cores and card; every result names the CPU | header; 2.1; evidence clause |
| Owner, third and fourth messages | engine code public; extending the owner's server weighed against a crate here by the measurements | 3; 4.7 |
| Decision 1 | model server stays, future measurement named | 4.1 |
| Decision 2 | no engine-side batcher | 4.6 |
| Decision 3 | decode threads default to physical cores, swept around it | 2.1; 4.4 |
| Decision 4 | the server is not a dependency; its nvJPEG bindings chosen, with the reason | 3; 4.7; 4.8 |
| Decision 5 | what N0 keeps of the pipeline, file by file | N0 keep table |
| Decision 6 | bars reframed (G1, G2, G3 to G9); end-to-end rate = min of both. Changed: on 20 MP the models-alone row is out of reach on one card because the decode's device work shares it, so G1 uses the card ceiling C, which equals the models-alone row on small photos | 0.2 |
| Decision 7 | card budget section; the detector's time from this repository's own rows (about 0.8 ms), not from the owner's server | 0a |
| Decision 8 | two-card row | N2 |
