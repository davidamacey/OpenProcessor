# Research plan: the best GPU image pipeline on this hardware (after B5)

| Field | Value |
|---|---|
| Status | plan, research. Nothing here is implemented. No code merges from this file without its experiment's pass line |
| Depends on | program B5 merged: `program/07a_b5_baseline.md` (5a reference `ref-b5a`, sweeps), `07b_b5_decode_once.md` (5b, incl. the 5b-P1 profile and the #407 stage records), `07c_b5_crops_ocr_faces.md` (5c), `07d_b5_processing_plans.md` (5d); then R-final and the v0.7.0 tag |
| Feeds | later milestones (section 5); B5's measurements decide which lines here open at all |
| Owner rules applied | library and documented pattern before any custom kernel; baseline before optimization; public data only (COCO, Open Images); run-to-run spread and parity gates on every claim; INT8 opt-in only; GPUs 0 and 2, GPU 1 only with the peer session's consent; GPU time in named windows, never per branch; decode once and keep frames resident |
| Sources (repo) | `docs/PERFORMANCE.md` (summary table, "Stage 2 tuning", "PE image encoder engine pass", "INT8 evaluation", "GPU decode measurements", "Decode once and the GPU decode", "0.7.0 stage profile", "GPU memory and the ingest load profile", "Scale ladder", Probe 3); `docs/research/v060_serving_backend_trade_study.md` (roofline, X-1..X-8); `docs/research/triton_deep_research_2026-09.md`; `docs/research/dali_gpu_decode_lessons.md`; `docs/research/v070_measurement_log.md`; `docs/design/triton_pipeline_optimization_plan.md`; `docs/design/gpu_decode_plan.md` (13); `docs/design/sp_gpu_prep_concurrency.md`; `docs/design/faces_ocr_resident_crops_plan.md`; `docs/design/processing_plans_plan.md`; `models/*/config.pbtxt`; `models/gpu_image_prep/1/`; issues #40, #42, #45, #67, #219, #220, #231, #268, #403, #407 |
| Sources (official docs, read 2026-11-04) | Triton python backend README (BLS, `async_exec`, decoupled, DLPack, one process per instance, GIL): https://github.com/triton-inference-server/python_backend ; Triton ensembles (`max_inflight_requests`, no `instance_group`): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html ; Triton dynamic batcher and priorities: .../user_guide/batcher.html ; model configuration (instance groups, warmup, rate limiter, CUDA graphs in `model_config.proto`): .../user_guide/model_configuration.html and https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto ; Triton backend API (C++ backends, `TRITONBACKEND_ModelServer` for in-process inference): https://github.com/triton-inference-server/backend ; Triton DALI backend: https://github.com/triton-inference-server/dali_backend ; DALI `fn.decoders.image` (`adjust_orientation` default true, `hybrid_huffman_threshold`, `jpeg_fancy_upsampling`, `hw_decoder_load`): https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html ; nvImageCodec (batched heterogeneous decode, ROI decode, `Encoder`, zero-copy to CV-CUDA/PyTorch/CuPy): https://docs.nvidia.com/cuda/nvimagecodec/ ; nvJPEG (backends, hardware decoder list): https://docs.nvidia.com/cuda/nvjpeg/index.html ; CV-CUDA 0.19 operators (`pillowresize`, `resize_crop_convert_reformat`, `copymakeborder`, `warp_affine`, `warp_perspective`, `laplacian`, `nms`, `ImageBatchVarShape`): https://cvcuda.github.io/CV-CUDA/ ; TensorRT (strongly typed networks, `trtexec`): https://docs.nvidia.com/deeplearning/tensorrt/latest/ ; NVIDIA Model Optimizer (`mtq.quantize` with `INT8_SMOOTHQUANT_CFG`, `mtq.auto_quantize` per-layer search): https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html and https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html ; CUDA MPS: https://docs.nvidia.com/deploy/mps/index.html ; Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html |

## 0. Summary

- **What B5 delivers, and where this plan starts.** B5 takes the public baseline on the final stack
  (5a), decodes each image once and evaluates CV-CUDA, DALI and nvImageCodec inside the image-prep
  model (5b), caps work crops and speeds up OCR recognition (5c), and runs faces and OCR as branches
  on the resident photo frame (5d). This plan does not repeat any of that. It starts from B5's
  records and asks what is still between this server and its hardware ceiling.
- **Where the time goes today.** Small images (COCO, about 0.3 MP) are bound by PE compute: about
  7.4 embeddings per image at 57 percent of the A6000's FP16 peak; one card does about 29 img/s, two
  about 55. Photos (12-24 MP) are bound upstream of PE: GPU decode, the Python image-prep executor
  (one GIL per instance) and device memory per instance. Faces, `/analyze` and OCR are bound by host
  CPU and by Python orchestration, not by their engines.
- **Top hypotheses.** H-1 a native (C++) resident-frame executor in Triton, if B5's profile shows
  Python, GIL and BLS overhead still bind; H-2 batched GPU decode for small images and stateless
  routes; H-5 per-card pipeline replicas on both A6000s; H-7 uint8 inputs and in-graph post-processing
  for the face models; H-9 a second, opt-in INT8 round for PE with SmoothQuant or a per-layer search.
- **GPU windows.** Three research windows after the v0.7.0 tag, about 30 GPU hours in total:
  RW-1 characterize (one card, about 9.5 h), RW-2 prototypes (one card, GPU 2 optional, about 11 h),
  RW-3 layout and confirmation (both A6000s, about 10 h).
- **Stop rule.** A line ends when its stage is not binding, when its prototype misses its bar or 2x
  the spread, when parity fails after one documented fix, or when it would need a second
  preprocessing implementation that cannot be gated forever (section 6).

## 1. Current measured state, per stage

Public numbers unless marked. One RTX A6000 unless marked. "Set A" is the pinned 2,000 COCO
images; "set B" the pinned 500 Commons photos (12-24 MP, about 1.6 detections per image). Numbers
are copied from the cited section, not re-derived. B5 retakes all of them on the final stack
(5a-R0); this plan uses B5's numbers once they exist and these only as context.

### 1.1 Per stage

| Stage | What runs today | Measured | Source |
|---|---|---|---|
| Decode, small images | CPU (Pillow/OpenCV, libjpeg-turbo), API process | 3.7-5.3 ms per image single thread; host is not the binder at 0.062 API CPU-s per image | PERFORMANCE "GPU decode measurements", summary table |
| Decode, photos | `gpu_image_prep`: nvJPEG GPU-Huffman lanes through ctypes, size gate 3 MP; progressive JPEGs (11 % of set B) fall back to the CPU | CPU decode 283-369 ms per photo; GPU route host CPU 0.34 CPU-s per photo (from 1.12); one nvImageCodec decoder saturates at about 35 photos/s; decode is 26-40 % of image-prep stage seconds | "Decode once and the GPU decode", `gpu_decode_plan.md` 13, "0.7.0 stage profile" |
| Resize and letterbox | CPU: Pillow BILINEAR (ingest), cv2 LetterBox (`/detect`); GPU: fused uint8 CuPy kernels | CPU letterbox 4.6 ms (COCO), about 80 ms (photo); GPU kernels about 1.3 ms GPU per 18 MP photo; letterbox 17.9 % of image-prep stage seconds at 4 batches | `triton_pipeline_optimization_plan.md` 11.3, `gpu_decode_plan.md` 13, "0.7.0 stage profile" |
| Detector | YOLO11s end2end TensorRT FP16, uint8 input, 2 instances | 1,583-1,712 img/s engine only at batch 16-64; about 2 % of ingest GPU time; NMS and head 3 % of the engine | "Stage 2 tuning" |
| PE embed (ingest) | PE-Core-L14-336 FP16 half-split engine, uint8 input, 1 instance | 229 img/s at batch 32 = 88 TFLOP/s = 57 % of 154.8 TFLOP/s; through Triton 217-226 img/s; 2nd instance +1-3 %; CUDA graphs -2 %; ingest set A 28.81-29.07 img/s (one card), 55.08-56.37 (two cards) | "PE image encoder engine pass", "Stage 2 tuning", "Multi-GPU PE encoding" |
| Photo ingest end to end | `gpu_image_prep` (Python BLS, decoupled, per-image worker threads, 8.7 GB per instance) | set B 35.0 img/s detect + embed, 44.0 detect only (0.6.0); 0.7.0 stage profile 26.64 at 4 batches (noisy host); two instances raise a non-public photo set from 31.8 to 48.0 img/s, a third adds 0.6 % | "Decode once and the GPU decode", "0.7.0 stage profile", "GPU memory and the ingest load profile" (ratio only) |
| Faces | CPU decode, SCRFD FP16 engine with FP32 640x640 input, anchor decode and NMS in Python, host `cv2.warpAffine` alignment, ArcFace FP16 with FP32 input | `/faces/recognize` 179.3 img/s set A, 36.8 set B; API CPU 0.047 / 0.364 s per image; SCRFD and ArcFace about 1-2 ms Triton exec per image | Probe 3, `processing_plans_plan.md` 2.2 |
| Embed (`/embed/image`) | MobileCLIP2-S2 FP16, FP32 256x256 input, CUDA graphs off, 25 ms queue delay | 223.0 img/s set A, 40.2 set B; inside `/analyze` MobileCLIP is the largest GPU item (11.4 ms exec per image at 16 clients) | Probe 3 |
| `/analyze` | API-orchestrated detector + faces + MobileCLIP, repeated decodes | 55.5 img/s set A, 22.8 set B; API CPU 0.116 / 0.789 s per image | Probe 3 |
| OCR | `ocr_pipeline` Python BLS, uint8 inputs (#220 step 1), one blocking recognition call per line | 20.2 img/s set A, 1.2 set B (before uint8); BLS 45-48 ms per COCO image for about 9 ms of GPU; width buckets rejected (P-ocr 86 %) | Probe 3, #220 comment 2026-11-02 |
| Write path | coalesced bulk writes, Lucene HNSW | write ceiling 4,426 documents/s with 8 writers (about 740 needed at the target); 42 % of API stage time at one batch on set A (latency, not throughput) | "Stage 4", "0.7.0 stage profile" |
| Cluster pass | CPU IVF after ingest | 2,042 s at 490,804 crops | "Scale ladder" (out of scope here: GPU clustering is its own agreement-gated track) |

### 1.2 Where the time goes

| Workload | Binding resource | Evidence |
|---|---|---|
| Small images, policy `all` | PE tensor compute on each card | encoder 95-97 % busy; 7.4 PE calls per image; 99 % of per-image FLOPs are PE (trade study 1.3) |
| Small images, detect only or `/detect` | host CPU and the client side (decode, letterbox, gRPC) | detect-only 120-132 img/s with the GPU far from busy; forcing small images onto the GPU route lost 21-49 % (per-image Python overhead) |
| Photos, ingest | image-prep executor: decode, one Python process per instance, device memory | GPU 55-94 % busy depending on instances and load list; +51 % from a second process; fallbacks to CPU decode when the card fills (`sp_gpu_prep_concurrency.md` 1) |
| Faces, `/analyze` | host CPU: CPU decode, FP32 tensors, Python post-processing and alignment | API CPU 0.36-0.79 s per photo against a few ms of GPU |
| OCR | Python BLS orchestration, sequential recognition | 45-48 ms BLS per image for about 9 ms GPU |

### 1.3 Roofline ceilings per card (estimates from measured costs; the pipeline ceiling is the minimum)

| Resource | Basis | Ceiling, one A6000 | Binds? |
|---|---|---:|---|
| PE FP16 at 57 % of peak | 229 embeddings/s; 7.4 per COCO image; about 2.6 per set B photo (1.6 detections + frame) | about 31 COCO img/s; about 88 photos/s | COCO: yes, now. Photos: not yet |
| PE at the best published TensorRT ViT-L fraction | trade study 2.2: about 50 % on this GPU; PE is already above it | no headroom from engine tuning alone | closed (PE engine pass) |
| PE INT8, if a recipe passed the gates | 1.2-1.4x measured for A8W8 recipes that failed the gates | about 37-43 COCO img/s | only if H-9 passes |
| Detector | 1,600-1,700 img/s at batch 16-64 | far above every other stage | no |
| GPU JPEG decode, photos | no hardware JPEG decoder on GA102 (nvJPEG docs; trade study 3.4); one nvImageCodec decoder about 35 photos/s; GPU-Huffman lanes reach 48 photos/s with two processes (non-public set, ratio) | unknown upper bound: E-2 measures it | yes, with the executor |
| Host CPU | 0.34 CPU-s per photo (GPU route), 0.062 per COCO image; about 24 physical cores shared with OpenSearch | about 70-140 photos/s; several hundred COCO img/s | not at one card; possibly at two cards on photos |
| PCIe 3.0 x16 (Haswell root complex) | about 12 GB/s achievable; JPEG in, kilobytes out | over 1,000 photos/s | no |
| Device memory | default load list holds 36.5 GB; 8.7 GB per image-prep instance | two instances fit only with the ingest load profile | yes, it caps concurrency |
| OpenSearch writes | 4,426 docs/s (8 writers) | about 600 COCO img/s equivalent | no |

Reading: for small images the only large levers left are the second card (about 2x) and precision
(opt-in, gated). For photos the gap to the PE ceiling is about 2x on one card (35-48 measured
against about 88), and it sits in decode, the executor and memory. For faces, `/analyze` and OCR the
gap is host and Python overhead, many times the GPU cost.

## 2. What B5 owns, and what this plan reads from it

| B5 task | What it decides here |
|---|---|
| 5a-R0, 5a-R1 (`ref-b5a`) | the reference for every experiment below; retaken at R-final for the shipped stack |
| 5a-S1, 5a-S3 two-card sweeps | the starting point for H-5 (replicas); if they already reach 1.8x on set B, H-5 shrinks to the NUMA and routing arms |
| 5a-S4 image-prep instances, nvJPEG at 20 MP | the decode ceiling input for H-1 and E-2 |
| 5a-S5 detector CUDA graphs, 5a-S6 `priority_levels`, 5a-S8 warmup and pools | H-8 covers only the engines 5a-S5 did not; H-11 starts from 5a-S6's outcome |
| 5b-P1 profile, 5b-G2 (CV-CUDA vs CuPy), 5b-G3 (DALI decode and letterbox), 5b-G4 rule (native backend) | the decider for H-1: if 5b-G4's rule fires, its required plan file is written from section 3 H-1 and E-7 of this plan |
| 5b-W1 Wave 6 bake-off on the stateless routes | the decider for H-2: a no-go there with a named binder opens or closes H-2 |
| 5b-C1 nvImageCodec crop encode and progressive decode | closes or keeps H-4's encode part |
| 5c-K1..K4 crop caps and reduced decode | closes H-12 (frames shared across processes) if worker re-decode becomes cheap |
| 5c-O1..O3 OCR `async_exec`, padded decode, argmax | the starting point for H-10 (OCR executor) |
| 5d-P5 faces and OCR as image-prep branches, CV-CUDA `warp_affine` | the starting point for H-7 on the GPU route; H-7's CPU-route part is new |
| #231 alternative-server head-to-head (deferred row of B5) | runs after 5b against the post-#407 Triton baseline with its own rule; not repeated here |
| #403 MPS (parked) | H-6 keeps its one-off check only |

## 3. Hypothesis register

Expected gains are estimates from the evidence column. "Decided by" names the B5 record or the
experiment that settles it. Every library row was checked against the official docs listed in the
header on 2026-11-04; re-read the pinned version's docs before coding.

| ID | Hypothesis | Evidence | Library or documented feature | Expected gain | Risk | Decided by |
|---|---|---|---|---|---|---|
| H-1 | A native resident-frame executor (C++ Triton backend) beats the Python image-prep model on photos: no GIL, no stub process, no per-image Python | one instance is one Python process whose threads share a GIL (python backend README); a second process gave +51 %, a third +0.6 %; Triton CPU 610-865 %; small images on the GPU route lost 21-49 %; `ocr_pipeline` spends 45 ms of Python per 9 ms of GPU | Triton backend API (C++); in-process inference to the TensorRT models through `TRITONBACKEND_ModelServer` and the server C API (the python backend does BLS this way); nvImageCodec / nvJPEG C API; CV-CUDA C++ operators (`pillowresize`, `resize_crop_convert_reformat`, `copymakeborder`, `laplacian`) | the 2-instance throughput from one process with about half the device memory; up to the PE ceiling on photos (about 88/s) if decode keeps up | high: C++ code in the Triton image, build and ABI pinned to the Triton release, parity with today's kernels | 5b-P1 repeated after 5b-G2 (5b-G4 rule), then E-3 and E-7 |
| H-2 | Batched GPU decode makes the GPU route win for small images and for the stateless routes (`/detect`, `/embed/image`, `/faces/*`, `/analyze`) | nvImageCodec batch-32 decode 2,346 COCO img/s at 2.8 ms host CPU; DALI dynamic-batches inside Triton (average batch 3-8, lessons doc 4); per-image Python made the GPU route slower for frames | Triton DALI backend with dynamic batching, `fn.decoders.image(device="mixed")`; nvImageCodec batched decode; or H-1's executor with a dynamic batcher in front | detect-only frames above the 132 img/s CPU route; host CPU per frame about -50 % | medium: DALI resize and pad traps (lessons doc 3); chroma and EXIF defaults | 5b-W1 bake-off, then E-8 |
| H-3 | The decode ceiling on GA102 is higher than one decoder instance shows, with batching and more lanes | one nvImageCodec decoder saturates at about 35 photos/s whatever its threads; GPU-Huffman lanes scale with processes | nvJPEG batched GPU-Huffman (`nvjpegDecodeBatched`), DALI `hybrid_huffman_threshold`, nvImageCodec decoder instances | names the true ceiling (needed by H-1, H-5) | low (measurement only) | E-2 |
| H-4 | The remaining host work on photos (crop JPEG encode, progressive JPEGs) moves to the GPU | crop JPEG encode 12 % of image-prep stage seconds at one batch, host threads; 11 % of set B is progressive | nvImageCodec `Encoder` and decoder (nvJPEG CUDA encode) | small, mostly host CPU | low | 5b-C1; reopened here only if 5b-C1 recorded "not adopted" with the reason "not binding" and E-3 later shows it binding |
| H-5 | Per-card pipeline replicas (one Triton per A6000, pinned to the card's NUMA node) scale photos and frames to about 1.9x | data-parallel PE reached 1.91x; GPU 0 and GPU 2 are on different sockets (`SYS`, trade study 1.2); the resident frame must stay on the card of its consumers, so the replica unit is the whole pipeline | Triton `instance_group { gpus }`; one server per GPU with `CUDA_VISIBLE_DEVICES` and a `cpuset`; client least-outstanding routing | about 1.8-1.9x per added A6000 | medium: memory per card, routing in a stateless API (B4) | 5a-S1/S3, then E-11 |
| H-6 | CUDA MPS lets several executor processes share one card without time-slicing | two image-prep processes are two CUDA contexts; MPS runs kernels from several processes concurrently (MPS docs) | CUDA MPS control daemon (needs host root; opt-in overlay only) | 0-10 %; zero if H-1 lands (one process, many streams) | medium: root daemon, fault coupling | E-6 (the #403 E8 one-off), only on E-3 evidence |
| H-7 | Faces get cheap: uint8 inputs with in-graph normalization for SCRFD and ArcFace, SCRFD anchor decode and NMS in the engine, batched GPU alignment | uint8 inputs cut the detector and PE request 4x and API CPU 47 % (#214); SCRFD returns about 1 MB of FP32 per image to Python; alignment is host `cv2.warpAffine` | `export/uint8_input.py` pattern (TensorRT `Cast` in the graph); TensorRT NMS layer or the `EfficientNMS` plugin as in the detector's end2end head; CV-CUDA `warp_affine` on `ImageBatchVarShape` | `/faces/*` and `/analyze` API CPU per image -30 % or more | low-medium: face boxes within 1 px, ArcFace cosine | 5d-P0 rows and 5d-P5, then E-4 (offline) and E-9 |
| H-8 | CUDA graphs help the small fixed-shape engines at batch 1 (interactive latency) | graphs help models with many small kernels (trade study 3.2); PE and detector lose or gain about 2 % | Triton `optimization { cuda { graphs: true graph_spec } }` (TensorRT backend only) | 0-10 % at batch 1 on SCRFD, ArcFace, MobileCLIP, OCR detector | low | E-5 (engines 5a-S5 did not cover) |
| H-9 | An opt-in INT8 PE recipe passes the precision gates | A8W8 recipes were 1.2-1.4x faster but failed every vector gate; weight-only passed with no speed (#213); ViT activations have outliers, which SmoothQuant targets | NVIDIA Model Optimizer `mtq.quantize` with `INT8_SMOOTHQUANT_CFG`; `mtq.auto_quantize` per-layer format search with FP16 as an allowed choice | 1.15-1.33x PE, opt-in only, never a default | high prior of failure; a new embedding space if it drifts | E-10 |
| H-10 | OCR stops paying Python per line once recognition is concurrent and the detector post-processing is batched | BLS 45 ms for 9 ms GPU; 5c makes calls concurrent but keeps one Python process | after 5c: H-1's executor hosting the OCR branch, or CV-CUDA `warp_perspective` for line crops | OCR img/s 2x on set A if Python binds after 5c | medium: P-ocr | 5c-O1..O3 records, then E-3 (OCR rows) |
| H-11 | Interactive requests keep low p99 while bulk jobs fill the card | one Triton serves both; B2 moves jobs to Celery, B4 makes the API stateless | Triton `priority_levels`, `default_priority_level`, per-priority queue policy; rate limiter resources; a reserved instance for interactive traffic | interactive p99 within 2x of idle under full bulk load | low | 5a-S6, then E-12 |
| H-12 | A decoded frame should outlive the request and be shared with the curation workers (VLM, segmenter, region stage) | workers run later in other processes; today they read the crop cache | CUDA IPC handles; or no sharing (work crops at model size, 5c) | likely none once 5c's caps land | high: lifetime, memory, multi-process state | 5c-K3 crop bytes and 5c-K4; closed without a GPU run if worker crop reads are not binding |
| H-13 | Smaller memory per in-flight frame lets two executors fit beside the default load list and lets small cards use the GPU route | 8.7 GB per instance; the default list leaves no room for two; #42 profile turns the GPU decode off under 16 GB | Triton `--cuda-memory-pool-byte-size`; BLS `preferred_memory`; one shared allocator in H-1's executor | the 2-instance rate on the default load list | low-medium | 5a-S8 pools, 5b-G2, then E-7's memory column |
| H-14 | A newer Triton and TensorRT (D-38, Triton 26.09, TensorRT 11.3) changes engine speed | TensorRT release notes list per-version changes; 11.0 listed 5-10 % CNN regressions | the pinned Triton image | -5 to +5 % | low | E-1 rows taken once after the image moves; not a research line of its own |

Not hypotheses (closed by evidence; do not reopen without new evidence): PE engine tuning by builder
flags, profiles or a second instance (PE engine pass); system shared memory and the gRPC ingest
channel (summary table); CUDA shared memory from the API (the API has no GPU; Triton known issue for
multithreaded clients, trade study 3.1); Triton's ONNX Runtime backend for GPU models (lessons doc 6);
DALI CPU decode inside Triton (lessons doc 5); 2:4 sparsity and FP8 (SM86, trade study 9); per-crop
BLS calls and chained ensembles for variable fan-out (`processing_plans_plan.md` 5.3).

## 4. Experiments

### 4.1 Common protocol

Every experiment follows `program/07_b5_overview.md` M-1..M-10 and section 6 gates (they move to
`docs/PERFORMANCE.md` "Benchmark protocol" when B5 closes; use that copy then). In short:

- isolated compose project `-p op-rw<n>-bench`, own ports, never the default project; GPU 0 (GPU 2
  only when the peer sessions agree; GPU 1 only with the owning peer session's written consent);
- host load, foreign GPU processes, GPU samples every 500 ms, clocks and power (`nvidia-smi dmon -s
  pucv`) recorded per run; a run with 1-minute load above 16 is noisy and repeated;
- 1 warm-up plus 3 counted repetitions for headline rows; spread = (max - min) / median; adopt only
  if the median gain beats 2x the larger spread and the parity gate passes;
- tiered: tier 1 one repetition on a 500-image slice to screen; tier 2 three repetitions for the
  candidate and its control; one full confirmation per bundle (E-13); one factor at a time;
- reference: `ref-b5a`, or the R-final capture if the shipped stack moved after 5a;
- raw data under the gitignored evidence directory; published numbers in `docs/PERFORMANCE.md`,
  method notes in `docs/research/`.

Parity gates used below (thresholds from `scripts/bench/parity_gates.json`):

| Gate | Rule |
|---|---|
| G-byte | byte-identical outputs where the math is unchanged (`gpu_prep_parity.py compare`) |
| G-tensor | model input tensor: best shift (0,0), mean abs diff < 0.5 grey, max <= 8 |
| G-box | every matched box within 1 px at the original resolution; P-box >= 0.98, P-item >= 0.99 |
| G-emb | PE crop and frame cosine median >= 0.999, p01 >= 0.995; P-knn overlap@10 >= 0.95; P-cluster-assign >= 0.98 |
| G-face | face boxes within 1 px; ArcFace cosine p01 >= 0.995; P-face 1.0 |
| G-ocr | P-ocr exact text on >= 98 % of matched lines; CER not worse |
| G-prec | the full `precision` profile of `parity.py compare` (the #213 table) |

Datasets (public only):

| Name | Content | Use |
|---|---|---|
| set A | 2,000 COCO (pinned) | frames, PE-bound rows, face probes on COCO person images |
| set B | 500 Commons photos 12-24 MP (pinned); the 2,000 extension if OQ-B5-1 created it | photo rows |
| COCO val2017 | 5,000 | AP and recall checks when boxes move |
| OI-10k, OI-100k | Open Images v7 pins (`openimages_bench_100000`, 10k slice) | scale legs in RW-3 |
| OI-faces | Open Images v7 images with the "Human face" box class, via 5a-A1's `--category people`, 2,000 pinned | face rows (COCO has no face boxes) |
| OCR set | the pinned text-bearing set from B0 and 5c | OCR rows |

The private high-res set is not used. The owner may run any experiment on it privately; such
results stay local and never appear in public text.

### 4.2 Experiment table

| ID | Tests | Protocol | Data and size | Metric | Parity gate | Pass bar | GPU h | Window |
|---|---|---|---|---|---|---|---:|---|
| E-1 | roofline sheet; H-14 | `perf_analyzer` per engine (PE, detector, SCRFD, ArcFace, MobileCLIP image, OCR det, OCR rec) at batch 1, 8 and max, through Triton; clocks and power logged; achieved TFLOP/s from the trade study's FLOP counts | synthetic tensors, plus 200 set A inputs for a real-input check | inferences/s, GPU ms per inference, achieved fraction of peak | none (no output change) | the sheet exists; any engine under 25 % of peak at its served batch is flagged for review | 2 | RW-1 |
| E-2 | H-3 | decoder matrix: today's GPU-Huffman lanes, nvImageCodec batched, DALI `mixed` (`jpeg_fancy_upsampling=True`, `adjust_orientation=True`) at batch 1, 8, 32, 64 and 1-4 decoder instances; reuse `scripts/bench/gpu_decode_bench.py` | set B 500 (2,000 if pinned), set A 2,000, EXIF fixtures 1-8 | photos/s, host CPU ms and GPU ms per image, device MiB | G-tensor against the CPU decoder; EXIF orientation equal on tags 1-8 | names the fastest gated decoder per size class and the photo decode ceiling on one card | 3 | RW-1 |
| E-3 | H-1, H-6, H-10 decider | Nsight Systems (CUDA API trace: launches, syncs, mallocs, copies, gaps) and py-spy on the image-prep model with every 5d branch on, and on `ocr_pipeline`; ingest profile with 1 and 2 instances; the 5b-P1 method on the post-B5 code | set B 500 at 4 and 8 batches; OCR set at 16 clients | share of the per-photo critical path in Python, GIL wait, BLS call overhead; GPU idle gaps; syncs and allocations per image | none | H-1 opens only if Python + GIL + BLS overhead >= 25 % of the critical path at the bulk point **and** GPU busy < 85 %; else H-1 stops | 2 | RW-1 |
| E-4 | H-7 offline | ONNX Runtime CPU equality of uint8-prefixed SCRFD and ArcFace against today's client normalization; SCRFD with anchor decode and NMS in the graph against today's Python post-processing | 500 OI-faces images, 200 set A person images | max abs diff of outputs; box equality | G-face offline | equality within FP32 rounding; no GPU needed | 0 | lane (CPU) |
| E-5 | H-8 | CUDA graphs with `graph_spec` per served batch for SCRFD, ArcFace, MobileCLIP image, OCR det; control is the same config without graphs | perf_analyzer, batch 1 and 8 | p50 latency at batch 1, throughput at batch 8 | G-byte | gain > 2x spread at batch 1 p50 or batch 8 throughput, p99 not worse; else record and close | 1.5 | RW-1 |
| E-6 | H-6 | the #403 E8 one-off: MPS overlay on a scratch stack, image-prep with 2 instances, control without | set B 500 at 4 and 8 batches | photos/s, GPU busy | G-byte | run only if E-3 shows two processes time-slicing (idle gaps while both are active); adopt as opt-in overlay only with >= 10 % gain | 1 | RW-1 |
| E-7 | H-1, H-13 | prototype C++ backend for the photo path's core: decode, letterbox, whole-frame square, detector and PE through in-process inference; the existing kernels ported or the CV-CUDA operators that 5b-G2 adopted; against the image-prep model at 1 and 2 instances, same flags (detect only; detect + embed frame) | set B 500 at 4, 8, 16 batches | photos/s, GPU busy, host CPU-s per photo, device MiB per instance, p99 | G-byte against the image-prep model where the kernels match; else G-tensor, G-box, G-emb against `ref-b5a` | >= 1.25x photos/s of the best image-prep configuration at equal or lower device memory, or equal photos/s at <= 50 % of its memory, p99 not worse than +10 %; else H-1 stops | 4 | RW-2 |
| E-8 | H-2 | small images on the GPU route: the 5b-W1 winning arm (DALI ensemble) or E-7's executor with a dynamic batcher in front, against the CPU route | set A 2,000 at 16, 32, 128 clients; plans detect-only, `/analyze` default, ingest `all` | img/s, host CPU-s per image, p99 | G-box, G-emb, COCO val2017 AP not lower | detect-only >= 1.3x the CPU route, or host CPU per image -50 % at equal img/s; `all` not lower; else the size gate stays | 2 | RW-2 |
| E-9 | H-7 | live: uint8 SCRFD and ArcFace with in-graph decode and NMS (from E-4) and CV-CUDA batched alignment, on the CPU route; the GPU route is 5d-P5's | OI-faces 2,000 and set A at 1 and 16 clients | `/faces/recognize` and `/analyze` img/s, API CPU-s per image, bytes per request | G-face | gain > 2x spread and API CPU -30 %; else record | 2 | RW-2 |
| E-10 | H-9 (opt-in) | the #213 protocol unchanged (calibration set disjoint from the pinned 2,000, per-GPU FP16 reference, same parity tool); recipes SmoothQuant INT8 and `auto_quantize` with FP16 allowed per layer; Q/DQ placement per the #213 notes | pinned 2,000 (12,790 crops + 2,000 frames) | PE embeddings/s through Triton at batch 16 and 32 | G-prec, every gate | every gate passes **and** >= 1.15x PE embeddings/s; else the INT8 line closes for good | 3 | RW-2 (GPU 2 if agreed, in parallel with E-7..E-9) |
| E-11 | H-5 | arms: (a) one Triton, every model and the executor with `instance_group { gpus: [0, 2] }`; (b) one Triton per card, `cpuset` on the card's NUMA node, the API routing least-outstanding across both | set A 2,000 and OI-10k; set B 2,000 (or 500 x 4 passes) | scaling against one card, p99, host CPU, memory per card | G-byte (same engines, same math) | >= 1.8x one card on set B and set A; pick by img/s, then p99, then operability; below 1.6x name the binder and stop | 4 | RW-3 |
| E-12 | H-11 | bulk ingest through Celery at full rate on both cards plus interactive `/detect` and `/faces/recognize` at 1-4 clients; arms: 5a-S6 setting (baseline), rate-limiter resources, a reserved interactive instance | set B bulk; set A interactive | interactive p99 against its idle p99; bulk img/s | none | interactive p99 <= 2x idle with bulk >= 95 % of its solo rate | 2 | RW-3 |
| E-13 | bundle | the adopted bundle against the R-final reference: headline rows, the full parity capture, one OI-100k leg with stop and resume; optional 3080 Ti row only with the peer's consent | set A, set B, OI-100k | img/s, GPU busy, memory, stage table | every gate that applies | no row below the reference by more than its spread; each adopted change keeps its own gain within 2x spread | 4 | RW-3 |

### 4.3 GPU windows

Three windows after the v0.7.0 tag. Each is announced to the peer sessions, runs one heavy job at a
time, and never overlaps a live-harness run or another window. A window that fails is re-run only
after the fix is merged, not per attempt.

| Window | Cards | Experiments | GPU hours | Entry condition |
|---|---|---|---:|---|
| RW-1 characterize | GPU 0 | E-1, E-2, E-3, E-5, E-6 (conditional) | about 9.5 | v0.7.0 tagged; B5 records in `docs/PERFORMANCE.md` |
| RW-2 prototypes | GPU 0; GPU 2 for E-10 if agreed | E-7 (if E-3 passed), E-8, E-9, E-10 | about 11 | RW-1 report reviewed; E-4 green offline; prototype lanes coded with T0 tests only |
| RW-3 layout and confirm | GPUs 0 and 2 together | E-11, E-12, E-13 | about 10 | RW-2 adoptions merged on the research integration branch; both cards agreed with the peers |

Optional riders in B5's GPU window 2 (owner's yes needed; zero code; under one hour in total; drop
them if the window is tight): the E-1 sheet for the engines 5a's perf_analyzer rows already load,
and a `dmon` clock and power log beside every 5a row. They save part of RW-1, nothing more.

## 5. Target architecture after the research

What it looks like if the main lines pass. Each box that depends on a hypothesis is marked; if that
hypothesis stops, the box keeps its B5 form.

```
API workers (stateless, B4) and Celery workers (B2)
  | JPEG bytes + resolved processing plan (5d); interactive at priority 1, bulk at priority 2 (H-11)
  | routed least-outstanding to one card (H-5)
  v
+-------------------- card 0 (NUMA node 0) --------------------+  +---- card 2 (NUMA node 1) ----+
| Triton server process (one per card, cpuset to its node)     |  | identical replica            |
|                                                              |  |                              |
|  front end: batched decode, uint8 frame resident in GPU      |  |                              |
|    memory, one decode per image, byte-budget admission       |  |                              |
|    (nvJPEG GPU-Huffman / nvImageCodec / DALI: E-2 picks)     |  |                              |
|                                                              |  |                              |
|  executor (native C++ backend if H-1 passes, else the        |  |                              |
|    Python image-prep model): one process, many CUDA streams, |  |                              |
|    per-request frame lifetime, in-process calls, no copies   |  |                              |
|    of the frame; views cut per consumer, batched:            |  |                              |
|      letterbox 640 uint8 ---------> detector (TensorRT)      |  |                              |
|      PE squares 336 (frame + N) --> PE (TensorRT, FP16;      |  |                              |
|                                     INT8 only opt-in, H-9)   |  |                              |
|      SCRFD 640 uint8 ------------> SCRFD with in-graph       |  |                              |
|                                     decode + NMS (H-7)       |  |                              |
|        warp_affine 112 (CV-CUDA) -> ArcFace (TensorRT)       |  |                              |
|      OCR det view -> line crops --> OCR rec with argmax (5c) |  |                              |
|      work crops <= crop_max_side -> GPU JPEG encode (H-4)    |  |                              |
|  outputs: boxes, vectors, strings, small JPEGs (kilobytes)   |  |                              |
|                                                              |  |                              |
|  engines owned by this server: every TensorRT model of the   |  |                              |
|  load profile, on this card only                             |  |                              |
+--------------------------------------------------------------+  +------------------------------+
  CPU route (same plan, same geometry functions): fallback for declined files, and the route for
  small images unless E-8 passes.
Curation workers (VLM, segmenter, region stage): read work-crop JPEGs from the crop cache; no frame
crosses a process boundary (H-12 closed).
```

Ownership and sharing rules:

| Rule | Why |
|---|---|
| The API owns no GPU memory and no GPU model | B4 statelessness; CUDA shared memory from multithreaded clients is a known Triton issue |
| One Triton server per card owns every engine on that card; no tensor crosses the socket | GPU 0 and GPU 2 are `SYS` apart; the resident frame must sit beside its consumers |
| The executor owns the frame for one request only | no cross-request or cross-process frame state (the multi-user rule: no in-process locks for shared state) |
| Request and batch paths share the same executor and engines; they differ only in priority and in-flight bounds | one implementation of preprocessing; batch throughput comes from many in-flight requests meeting in the dynamic batchers |
| Every branch is a flag of the resolved plan; an unrequested branch costs nothing | #67 acceptance, kept |

## 6. Stop rule

A line of research ends, is recorded in `docs/PERFORMANCE.md` or `docs/research/`, and its issue
closes, when any of these holds. Its code, settings and model directories are deleted in the same
change (no dead code, no switches left behind).

1. **Not binding.** Its decider shows the target stage under 10 % of the binding stage's time at
   the bulk point, or E-3's bar (25 % Python share and GPU under 85 % busy) is not met for H-1.
2. **No gain.** The tier-2 median gain is under 2x the spread, or under the line's own bar
   (section 4.2), at the target concurrency.
3. **Parity.** A gate fails, one documented fix is tried, and it fails again.
4. **Two implementations forever.** Adoption would need a second preprocessing path that cannot be
   held to G-byte or G-tensor against the first in every release (trade study rejection of
   architecture E).
5. **Budget.** The line needs a second unplanned GPU window, or more than twice its GPU hours.
6. **Supersede.** A cheaper line (for example the second card, H-5) delivers the same ceiling.

"Park" is not "stop": a line parked waits for named evidence (as #403 waits for E-3), and the issue
says which number would reopen it.

## 7. Sequencing after B5

Order: B5 (v0.7.0) -> R-final -> tag -> the research milestone below. The v0.8.0 platform-breadth
milestone (#42, Wave 8 region and segmenter options, Wave 10 with #365) runs in parallel where it
does not need a research window; Wave 8 option (b) keeps its own plan file per the owner's Q8 answer.

| Step | Content | Issue (existing or proposed) | Needs |
|---|---|---|---|
| S0 | publish the B5 numbers into this plan's section 1 (replace the context rows); decide which H-n are open | this file, docs commit | B5 closed |
| S1 | offline lanes (CPU, T0 only): E-4 face-graph equality; E-7 prototype skeleton and its offline tests; E-10 recipe scripts | proposed N-R2, N-R4, N-R5 | S0 |
| S2 | RW-1 | proposed **N-R1** `research(gpu): post-B5 characterization window (roofline sheet, decode ceiling, executor profile)`: E-1, E-2, E-3, E-5, E-6; one report in `docs/research/` | S0; peers agree |
| S3 | RW-2 | **N-R2** `research(gpu): native resident-frame executor prototype (C++ Triton backend)`: opens only on E-3's bar; brief: port decode, letterbox, whole frame, detector and PE to a C++ backend using nvImageCodec/nvJPEG and CV-CUDA, in-process inference; E-7 gates. **N-R3** `perf(gpu): batched GPU decode for small images and stateless routes`: E-8, after the #40 Wave 6 record. **N-R4** `perf(faces): uint8 inputs and in-graph SCRFD decode and NMS`: E-4 then E-9. **N-R5** `research(precision): opt-in INT8 PE, second round (SmoothQuant, per-layer search)`: E-10; closes the INT8 line if it fails | RW-1 report |
| S4 | adopt what passed, behind a switch until E-13 | each adopting PR on the research integration branch; `MODEL_VERSION` bumps; ATTRIBUTION rows for adopted libraries | S3 passes |
| S5 | RW-3 | **N-R6** `perf(gpu): per-card pipeline replicas and NUMA pinning on two A6000s`: E-11. **N-R7** `perf(serving): interactive latency under bulk Celery load`: E-12, after B2/B4 | S4 merged |
| S6 | E-13 confirmation, then defaults move; #45's 1M row may run here on the owner's word (extrapolate first) | #45 remainder; #40 epic closes if its children are closed | RW-3 |
| side | MPS only on E-3 evidence | #403 (parked) | E-3 |
| side | alternative-server head-to-head | #231 (its own rule, after 5b) | B5b |
| side | Triton 26.09 / TensorRT 11.3 rows | D-38, the image move; E-1 rows only | the image move |
| side | 12 GB card rows | #42 on GPU 1, peer consent | GPU 1 free |

Milestone: the old `v0.9.0` is closed once empty (`00_index.md` section 3). Proposed: a new
milestone for this research (name in OQ-1), holding N-R1..N-R7 and #403, #231 if B5 leaves them
open. Issue creation goes through the GitHub actions file of the active program, on the owner's go.

Plan lifecycle: this file is a working paper. When the research milestone closes, measured numbers
move to `docs/PERFORMANCE.md`, the as-built design to `docs/ARCHITECTURE.md` (GPU pipeline), method
notes to `docs/research/`, and this file is deleted in the closing PR.

## 8. Risks

| Risk | Mitigation |
|---|---|
| a native backend becomes a maintenance burden tied to each Triton release | E-7's bar is high (1.25x or half the memory); the backend reuses the Triton backend template and CV-CUDA/nvImageCodec, not hand-written kernels; stop rule 4 |
| custom kernels creep back in | every prototype lists the library operator it uses per step; a custom kernel needs a written reason in the PR (LIB-8) |
| peer load skews numbers on the shared 48-thread host | M-3 records; noisy runs repeated; windows announced; tiered tests |
| two-card runs take GPU 2 from another session | RW-3 is the only two-card window; ask first; E-11 can fall back to the 5a-S3 record |
| DALI or nvImageCodec defaults differ from the CPU decoder (chroma, EXIF) | G-tensor first; DALI's `adjust_orientation` now defaults to true, so the older "DALI ignores EXIF" note must be re-checked on the pinned version |
| INT8 drift slips into a default | opt-in only; G-prec; never in an installer profile |
| a gain on set B does not carry to dense street scenes | report detections per image beside every photo row; OI-10k leg in RW-3 |
| research outruns correctness work | nothing here starts before the v0.7.0 tag; no research PR touches the curation write path |

## 9. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| OQ-1 | Milestone for this research: reopen the closed `v0.9.0`, or a new name | a new milestone, for example `research-gpu-pipeline`, so old triage notes on `v0.9.0` stay meaningful |
| OQ-2 | Is a C++ Triton backend acceptable in the repo if E-7 passes (build toolchain in the Triton image, C++ review) | yes, only on E-7's bar, as its own plan file per 5b-G4 |
| OQ-3 | Riders in B5's GPU window 2 (section 4.3) | yes if the window has slack; otherwise all of it in RW-1 |
| OQ-4 | Faces benchmarks need face boxes: OK to pin an Open Images "Human face" subset (CC BY 2.0 images, CC BY 4.0 annotations) | yes, pinned manifest, images never committed |
| OQ-5 | INT8 second round (E-10): worth 3 GPU hours given the first round's wide misses | yes, once; it closes the line for good if it fails |
| OQ-6 | RW-3 needs both A6000s for about 10 hours: acceptable to schedule with the peer session | yes, one agreed day |
| OQ-7 | Should the research also cover fewer PE calls per image (skip near-duplicate or tiny crops)? It changes stored vectors, so it is a product rule, not a speed hack | no, not here; it belongs with the embedding-space migration (Wave 10, #365) |
| OQ-8 | A card with a hardware JPEG decoder (A100, H100 class) would change the photo ceiling; is any future hardware in scope | no; plan for GA102 only and say so in the results |
