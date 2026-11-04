# Research plan: GPU image processing at speed for the clean backend rebuild

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [program/03b](03d_stage4_job_families.md) (W2.3a and W2.3b), [program/03d](03b_stage2_control_data.md), [program/03c](03f_stage6_scheduler_scaleout.md), [program/05](05_b4_stateless_api.md) and [program/03g](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([program/00b](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([program/00f_research_track.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan, research. Nothing here is implemented or measured by this file. No product code merges from it without its prototype's pass line |
| Written | 2026-11-04 on `research/gpu-speed` (from `origin/main` at `497315954`), read-only: code, plans, issues and official docs read; no GPU job, no stack, no test suite run |
| Replaces | the earlier version of this file (post-B5 hypothesis register H-1..H-14, experiments E-1..E-13, windows RW-1..RW-3; in git history). Every hypothesis and experiment of it is carried below, re-ranked against the measured data, and re-mapped to prototypes PR-n (section 6.9 maps old to new IDs) |
| Owner direction applied | 2026-11-04: the backend is rebuilt clean on PostgreSQL, Celery and Valkey (D-39..D-69); the GPU work "will materially change how we do the GPU processing at speed", so it is researched to best practice first; simple prototypes run in a local Python venv on the GPU, no Docker image builds; datasets are set P (private, about 21 MP), the public image sets and public vector sets where relevant; plan first, run nothing now |
| Standing rules applied | documented library before custom code ([standards_and_libraries.md](reference/standards_and_libraries.md)); baseline before optimization (#45); parity gates and 2x-spread rule on every claim ([06_b5_overview.md](06_b5_overview.md) sections 5-6); INT8 opt-in only; GPUs 0 and 2 only, one GPU job at a time, peers asked first; no dead code or switches left behind |
| Feeds | the rebuild seams (section 5) into [02a_postgres_schema.md](02a_postgres_schema.md), [02b_jobs_locking_events.md](02b_jobs_locking_events.md), [03e_stage5_gpu_queues.md](03e_stage5_gpu_queues.md), [05_b4_stateless_api.md](05_b4_stateless_api.md); the B5 sub-batches [06a](06a_b5_baseline.md)-[06d](06d_b5_processing_plans.md); the post-tag research wave. Section 11 lists every change the integration agent must make in those files |
| Issues | #40 (epic), #407, #219, #220, #268, #403, #231, #42, #45, #67 |

Plain-words rule: every heading names the thing first and the code after it, for example "the
decoder matrix (prototype PR-1)".

---

## 1. Plain-language summary: what changes and why it matters

**What the backend does with an image today.** A photo arrives as JPEG bytes. Small images (COCO
size, about 0.3 MP) are decoded on the API's CPU and sent to Triton as small uint8 tensors. Large
photos (3 MP and up) are sent as bytes to one Triton Python model, the image-prep model
(`gpu_image_prep`), which decodes them on the GPU once, cuts every model input from that frame
(detector letterbox, whole-frame square, one square per detected object), calls the detector and
the PE embedding model inside Triton, and returns boxes, vectors and small crop JPEGs. Faces, OCR
and `/analyze` still decode on the CPU and send float tensors.

**What the measurements say limits speed.**

- On small images the PE image encoder is the limit: one A6000 is 95-97 % busy at about 29
  images/s, and 99 % of the arithmetic per image is PE. Only a second card (measured 1.91x) or a
  lower precision (opt-in, failed its gates once) moves that.
- On 12-24 MP photos the limit moved twice in 0.7.0: first device memory (the card was full and
  photos silently fell back to the CPU), then, after the ingest load profile freed 27.7 GB, the GPU
  itself: about 48 photos/s with the card 94-97 % "busy". PE is about 15 ms of that card's 21 ms
  per photo; decode, resize kernels, the detector and copies share the rest.
- Faces, OCR and `/analyze` are limited by host CPU and Python, not by their engines (a few ms of
  GPU per image against 36-780 ms of host work per photo).

**What this plan changes.**

1. **Measure the right GPU number.** "GPU busy" from `nvidia-smi` only says a kernel was running;
   it does not say how much of the card worked. Every prototype records DCGM profiling metrics
   (SM active, tensor pipe active) and an Nsight Systems trace, so "the GPU is the limit" becomes a
   per-photo GPU-millisecond budget per stage. That budget decides everything else.
2. **Pick the decoder, resize and crop libraries by measurement, in a venv, before the rebuild
   codes them.** nvJPEG (through today's ctypes lanes), nvImageCodec, DALI and CV-CUDA are compared
   on the same photos for host milliseconds, GPU milliseconds, memory and pixel parity. The official
   docs read for this plan changed three assumptions (section 3.5): nvJPEG's batched GPU Huffman
   decode needs batches over 50 images, its scaled and batched-ROI decode exist only on cards with a
   hardware JPEG decoder (not the A6000), and DALI's chroma upsampling default does not match
   libjpeg.
3. **Keep one implementation of every preprocessing step for the request path and the batch path**,
   hosted where the engines are (Triton), and make its contract data: each engine's input contract
   (dtype, size, resize kind, interpolation, fill, normalization) becomes a model-registry record, so
   the CPU route, the GPU route and the prototypes derive their inputs from one definition.
4. **Make the rebuild ready for the winners without a later schema change**: per-card serving
   (one Triton per card, chosen by a router), GPU identity by UUID (CUDA and `nvidia-smi` number the
   cards differently), model-registry fields for input contracts, build profiles and measured
   memory, a deployment serving profile as a revisioned config document, and a per-job performance
   summary. Section 5 lists each with its owner.
5. **Prototype first, then land.** Ten small prototypes (section 6.9) in a tracked, exactly pinned
   research folder answer the open questions with numbers; each has a pass bar and a stop rule. The
   B5 sub-batches then adopt only what passed.

**Why it matters.** The two shipped levers that remain large are the second card (about 1.9x) and
not wasting the GPU on work that is not PE. Everything else is in single-digit to tens of percent,
and the rebuild is the cheapest moment to put the seams in place: adding a column or a config kind
to the first Alembic revision costs minutes; adding it after the release costs a migration, an
upgrade step and a test of the upgrade.

## 2. What is measured today (every number with its source)

All numbers are copied from the cited section, not re-derived. One RTX A6000 unless marked. Set A
is the pinned 2,000-image COCO set (`coco_bench_2000`, seed 20261009); set B the pinned 500
Wikimedia Commons photos, 12-24 MP (`commons_hires_500`, seed 20261010); set P the owner's private
photo set (about 21 MP). Set-P numbers below were already published in the cited public log before
this plan; from now on new set-P numbers stay private (section 6.8).

### 2.1 Decode cost per image

CPU decoders, single thread, ms per image, median (Pillow 12.3, OpenCV 4.10 of the API image);
source: [PERFORMANCE "GPU decode measurements"](../../PERFORMANCE.md#gpu-decode-measurements-217),
[gpu_decode_plan.md](../gpu_decode_plan.md) 5.1.

| Images (median MP) | Pillow + EXIF | Pillow at 1/8 | OpenCV | OpenCV at 1/8 | both decodes ingest ran | both, 46 processes: images/s |
|---|---:|---:|---:|---:|---:|---:|
| COCO set A (0.27) | 5.3 | 2.2 | 3.7 | 2.1 | 7.4 | 979 |
| Open Images 1,024 px (0.78) | 10.1 | 3.4 | 5.8 | 2.9 | 13.0 | 851 |
| Open Images originals (5.2) | 68.1 | 15.5 | 37.1 | 15.6 | 83.7 | 124 |
| photos, set B (17.7) | 369 | 130 | 283 | 133 | 502 | 26.3 |

GPU decoders, batch 32, 8 decoder threads (images/s, host CPU ms per image, GPU busy ms per image);
source: [gpu_decode_plan.md](../gpu_decode_plan.md) 5.2.

| Class | nvImageCodec | DALI `mixed` | torchvision nvJPEG |
|---|---|---|---|
| A (0.27 MP) | 2,346; 2.8; 0.15 | 1,941; 3.1 | 379; 2.6 (serial) |
| OI (0.78 MP) | 1,282; 4.9; 0.26 | 1,176; 5.1 | 215; 4.6 |
| OIO (5.2 MP) | 193; 21.6; 1.0 | 154; 23.5 | 22; 45 |
| B (17.7 MP) | 35.4; 134; 3.9 | 31.8; 132 | 4.5; 221 |

- nvImageCodec 0.7.0.49's `GPU_ONLY` backend returned no image on the A6000; its working backend
  ran the Huffman stage on the CPU, about 7.5 ms of CPU per MP; batch 1 and 8 on set B gave 7.4 and
  13.8 photos/s at 135-152 ms host CPU; one decoder saturates at about 35 photos/s (threads 2: 13.5,
  4: 17.9, 8: 35.4, 16: 35.9, 32: 29.6; at most 5 cores) ([gpu_decode_plan.md](../gpu_decode_plan.md) 5.2).
- Decode plus model inputs on the GPU: `nvimgcodec_full` (decode + float32 torch kernels) A 446,
  OI 350, OIO 99, B 25.2 images/s (16.8 GPU ms per photo); `dali_full` (decode, letterbox, whole
  frame in DALI) A 1,592, OI 1,030, OIO 166, B 33.4 (3.8 GPU ms per photo) (5.3).
- Contention with PE on the same card: float32 torch kernels cost PE 22 % (14.6 ms of PE per
  photo); DALI 4.5-5.8 % (2.1 ms) (5.4). PCIe: 10.4 MB of JPEG per photo against 54 MB of RGB (5.2x
  less); COCO 168 KB against 0.85 MB (5.4).
- GPU memory per process: torch CUDA context 310 MiB; plus nvImageCodec holding 1 / 8 / 16 photos
  512 / 2,000 / 2,784 MiB (5.5).
- The shipped decoder: nvJPEG's decoupled API with the `GPU_HYBRID` backend (GPU Huffman) through
  ctypes, 16 lanes, libjpeg-style chroma interpolation (`models/gpu_image_prep/1/nvjpeg_lanes.py`),
  chosen because nvImageCodec decoded the Huffman stage on the CPU ([gpu_decode_plan.md](../gpu_decode_plan.md) 13).
- Coverage: nvImageCodec applies all 8 EXIF orientations and with `fancy_upsampling=1` matches
  Pillow within 0.01 grey on average; torchvision's GPU decode and DALI's default ignored EXIF then;
  every GPU decoder returned an image for truncated or corrupt files Pillow rejects
  ([PERFORMANCE "GPU decode measurements"](../../PERFORMANCE.md#gpu-decode-measurements-217)).
  11 % of set B is progressive and goes to the CPU (47 of 400 measured in SP0, plus 2 truncated)
  ([v070_measurement_log.md](../../research/v070_measurement_log.md) SP0.1).

### 2.2 Photo ingest end to end (the GPU route)

| Measurement | Value | Source |
|---|---|---|
| Set B detect + embed, batch 8 x 16 clients: CPU path / CPU single decode / + libjpeg-turbo / GPU decode | 27.0 (25.2-27.2) / 26.8 / 23.5 / **35.0 (24.2-38.5)** images/s | [gpu_decode_plan.md](../gpu_decode_plan.md) 13 |
| Set B detect only | 28.3 / 31.6 / 31.0 / **44.0 (38.8-44.8)** | same |
| One client, batch 1 (images/s, p50 s) | 0.97, 0.98 / - / 1.17, 0.81 / **2.40, 0.36** | same |
| Four clients, batch 1 | 3.45, 1.07 / - / 4.48, 0.85 / **8.30, 0.41** | same |
| Host CPU-s per photo at bulk (API + Triton) | 1.12 / 0.97 / 1.00 / **0.34** | same |
| GPU busy at bulk detect + embed | 35 / 33 / 28 / 55 % | same |
| Set A forced onto the GPU decode | detect + embed 22.7 against 29.0; detect only 63.6 against 124.5 (so the 3 MP gate) | same |
| PE beside the GPU decode | 223 to 213 inferences/s (-4.5 %) | same |
| Fused uint8 CuPy kernels | about 1.3 ms of GPU per 18 MP photo | same |
| Batching requests inside the model | a barrier between stages, 15 photos/s; per-image workers whose BLS calls meet in the models' batchers: 43-48 | same |
| Image-prep instances (3,275 photos, about 18 MP) | 1: 31.8; 2: 48.0 (+51 %, GPU 70-75 to 78-87 %, Triton CPU 610 to 865 %); 3: +0.6 % for 6-7 GB more | [PERFORMANCE "Decode once and the GPU decode"](../../PERFORMANCE.md#decode-once-and-the-gpu-decode-217); log E4 |
| E4 detail | count 1: 31.75 / 31.96 / 31.63; count 2: 47.06 / 49.16 / 48.01; count 3: 48.28 / 48.48 / 48.04; memory after 30.4-30.7 k / 34.6-37.9 k / 43.5-44.9 k MiB | [log E4](../../research/v070_measurement_log.md#e4-268-gpu_image_prep-instance-count-3000-distinct-nvme-photos-concurrency-16-batch-16) |
| 2-instance parity (200 COCO) | identical status, size, letterbox, scores, classes; embeddings and boxes within the run-to-run noise of count 1 (whole-embedding cosine 0.99992) | log E4 |
| Memory per instance | 8,704 MiB budgeted; stubs 268 MiB idle, 8.5-9.1 GB peak with headroom, 6.9-7.5 GB on a full card | [PERFORMANCE "Decode once..."](../../PERFORMANCE.md#decode-once-and-the-gpu-decode-217); log SP0.7.1 |
| Default load list | `tritonserver` 36,486 MiB with every default model; 8,864-8,896 MiB with the eight unused unloaded (27.6-27.7 GB freed); OCR recognizer alone 11,572 MiB | log SP0.7.1, SPF.3 |
| Per-model memory over the ingest profile (8,813 MiB) | SCRFD 6,846; ArcFace 2,754; MobileCLIP image 3,454; MobileCLIP text 1,430; OCR det 1,642; OCR rec 11,572; `ocr_pipeline` 0 MiB | log SPF.3 |

Stage profile, 0.7.0 code, Triton 2.70.0 (26.06), 2 instances, policy `all`; source:
[PERFORMANCE "0.7.0 stage profile"](../../PERFORMANCE.md#070-stage-profile) and
[log SP0](../../research/v070_measurement_log.md#sp0-stage-profile-ship-track-plan-section-5-perfv070-ingest-speed).

| Row | img/s median (min-max) | GPU util mean / p50 % | API CPU-s per image | Largest stages |
|---|---|---|---|---|
| Set A, 1 batch (CPU route) | 13.14 (13.12-13.47) | 45 / 30 | 0.078 | PE crop embed 54.5 %, OpenSearch write 42.1 % of API stage time |
| Set B, 1 batch | 9.88 (9.63-9.96) | 15 / 0 | 0.262 | decode 39.5 %, PE wait 22.3 %, detect 15.5 %, crop JPEG 12.0 % |
| Set B, 4 batches | 26.64 (25.21-29.00) | 49 / 43 | 0.303 | decode 26.2 %, letterbox 17.9 %, PE wait 15.4 %, detect 8.9 % |
| Set P, 1 / 4 / 8 batches | 14.80 / 24.98 / 20.13 | 35 / 58 / 39 | 0.078 / 0.200 / 0.701 | see the log's per-stage table |

The fall past 4 batches was device memory, not CUDA calls (log SP0.7): with the eight unused models
unloaded ("headroom") set P ran 48.5 img/s at 8 batches (GPU 96 %, 0 fallbacks) and 38.5 at 4;
on the full list 40.4 and 30.7 with 0.6-23 % of photos falling back to the CPU on CuPy and nvJPEG
out-of-memory errors, growing with session age. Thread sampling found no saturated thread (3.0-4.5
cores per stub, busiest thread 20-25 %): **the Python process was not GIL-bound at that load**
(SP0.7.5; py-spy not used). API CPU without fallbacks: 0.088, 0.084, 0.064, 0.062, 0.066 CPU-s per
image at 1, 2, 4, 8, 12 batches (SP0.7.6). One instance caps near 32 img/s with the GPU 75 % busy;
two reach about 48 (SP0.7.3). `WORKERS` 16 or `MAX_IN_FLIGHT` 16 gave +13 to +17 % at 4 batches;
LANES, ENCODE_THREADS and the shm sizes nothing (SP0.7.4).

Acceptance after the fixes (set P, config 16 workers / 32 in flight, host load 13-35; log SPF.4):

| Stack | Clients | img/s median (min-max) | GPU % | Fallback % | API CPU-s/img | Card peak GiB |
|---|---|---|---|---|---|---|
| ingest profile, 2 instances | 4 | 48.3 (45.3-49.0) | 94 | 0 | 0.052 | 25.4 |
| ingest profile, 2 instances | 8 | 48.3 (40.8-48.8) | 93 | 0 | 0.056 | 25.8 |
| default list, 1 instance | 4 | 38.9 (32.9-39.4) | 83 | 0 | 0.060 | 45.3 |
| default list, 1 instance | 8 | 38.6 (38.4-40.2) | 86 | 0 | 0.059 | 44.9 |
| default list, 2 instances | 4 | 35.9 (32.4-39.3) | 72 | 0.6 / 2.2 / 5.6 | 0.081 | 47.4 |
| 24 GiB emulation, ingest profile, 1 instance | 4 | 38.1 (35.1-39.1) | 84 | 0 | 0.059 | 42.1 with ballast |

The budget wait is not binding on the ingest profile (0.2 s over 8,976 photos at 16 workers)
(SPF.1). Embed reprocess pass on the GPU route (set P, log F1-G5): 28.1 img/s before the chunk
window, 51.12 (48.16-51.84) after `EMBED_CHUNKS_IN_FLIGHT = 2` at GPU 84.5-92.1 %, API CPU 0.025;
vectors against ingest p01 0.99993, median 0.999997; region vectors after the CPU path stopped the
JPEG round trip agree at p01 0.997 (G5, G6).

### 2.3 Engines

| Measurement | Value | Source |
|---|---|---|
| PE image encoder, FP16, half-split rotary layout | trtexec 229 img/s at batch 32 = 88 TFLOP/s = 57 % of 154.8 TFLOP/s; through Triton 217-226 (1 x 32: 217.0); 2nd instance +1 to +3 %; CUDA graphs -2 %; builder flags, profiles, opt shapes within 2 % except at the opt batch | [PERFORMANCE "PE image encoder engine pass"](../../PERFORMANCE.md#pe-image-encoder-engine-pass-227) |
| PE engine before | 13.7 ms to 5.5 ms per embedding (FP16) | [PERFORMANCE summary](../../PERFORMANCE.md#summary-of-the-optimization-milestone) |
| PE INT8 (#213) | A8W8 recipes 1.2-1.4x faster but fail the vector gates; weight-only passes with no speed | [PERFORMANCE "INT8 evaluation"](../../PERFORMANCE.md#int8-evaluation-of-the-pe-image-encoder-on-ampere-213) |
| Detector YOLO11s end2end FP16 uint8 | batch 1 / 8 / 16 / 32 / 64: 640 / 1,464 / 1,595 / 1,685 / 1,712 img/s; NMS plugin 0.09 ms of 10.1 ms per batch 16 (1 %); head glue 2 %; engine about 2 % of ingest GPU time; variants within the 1 % build-to-build spread | [PERFORMANCE "Stage 2 tuning"](../../PERFORMANCE.md#stage-2-tuning-detector-engine-and-the-concurrency-sweep-226-259) |
| Ingest set A, one card | 23.39 to 28.81-29.07 img/s, encoder 95-97 % busy; 16 x 16 best; Triton arms within -0.5 to +1.9 %; API workers 16-48 within the noise | same |
| Two A6000, data-parallel PE | 44.60 img/s (1.91x of 23.39); A6000 + 3080 Ti 43.72 (1.87x); tuned 55.08-56.37 | [PERFORMANCE "Multi-GPU PE encoding"](../../PERFORMANCE.md#multi-gpu-pe-encoding-228) |
| Request bytes per image, set A | uint8 inputs 14.94 to 3.73 MB; API CPU 0.118 to 0.062 CPU-s | [PERFORMANCE "uint8 inputs"](../../PERFORMANCE.md#wp-21-uint8-inputs-for-the-detector-and-the-pe-image-encoder-214) |
| System shared memory API to Triton | not adopted: under 5 % of API CPU | [PERFORMANCE "WP-2.2"](../../PERFORMANCE.md#wp-22-system-shared-memory-between-the-api-and-triton-215-measured-and-not-adopted) |
| Model FLOPs per inference | PE 384.4 GFLOP; YOLO11s 21.6; SCRFD-10G 26.7; ArcFace R50 12.6; MobileCLIP2-S2 image 15.7; per image at policy `all` (7.39 PE + 1 detector) 2,862 GFLOP, 99.2 % PE | [v060_serving_backend_trade_study.md](../../research/v060_serving_backend_trade_study.md) 1.3 |

### 2.4 Faces, OCR, `/analyze`, `/embed/image` (Probe 3)

Source: [PERFORMANCE "Probe 3"](../../PERFORMANCE.md#probe-3-faces-and-ocr-the-stages-ingest-does-not-call), 16 clients.

| Endpoint | Set A img/s | Set A API CPU-s/img | Set B img/s | Set B API CPU-s/img | GPU exec ms per image |
|---|---:|---:|---:|---:|---|
| `/detect` | 133.2 | 0.050 | 33.7 | 0.376 | detector 1.4 (A), 3.0 (B) |
| `/faces/recognize` | 179.3 | 0.047 | 36.8 | 0.364 | ArcFace 1.2 + SCRFD 1.0 (A) |
| `/embed/image` | 223.0 | 0.027 | 40.2 | 0.299 | MobileCLIP 1.7 (A), 4.5 (B) |
| `/analyze` | 55.5 | 0.116 | 22.8 | 0.789 | MobileCLIP 11.4 the largest (A) |
| `/analyze` with OCR | 20.0 | 0.160 | 1.2 | 2.755 | `ocr_pipeline` 45.9 (A), 765.8 (B) |
| `/ocr/predict` | 20.2 | 0.079 | 1.2 | 2.139 | `ocr_pipeline` 45.8 for about 9 ms of det + rec |

`ocr_pipeline` instances scale it linearly: 1 / 2 / 4 instances 21.4 / 42.6 / 80.0 img/s on
`/ocr/predict` (same source). OCR width buckets (#220 step 2, 1,500 COCO val2017, 440 regions):
padding to buckets P-ocr 86.0 % (char edit rate 1.94 %, BLS 43.2 ms); exact-width grouping 100 % at
49.5 ms against 48.0 (no gain); rejected (#220 comment 2026-11-02). The OCR inputs are uint8 since
#220 step 1 (`27ebeb45`). SCRFD and ArcFace still take FP32 inputs, SCRFD returns raw anchor
outputs decoded in Python, and alignment is host `cv2.warpAffine` (`models/scrfd_10g_bnkps/config.pbtxt`,
`src/utils/face_align.py`).

### 2.5 Scale and serving references

| Measurement | Value | Source |
|---|---|---|
| 100k ladder | 0.7 MP images GPU-embed bound (94 % mean GPU); 20 MP photos 21.0 img/s with the GPU about 51 % busy (27.2 on a 20k re-check) before the 0.7.0 fixes | #40, #45; [PERFORMANCE "Scale ladder"](../../PERFORMANCE.md#scale-ladder-10k-and-100k) |
| Source storage | not on the critical path: HDD RAID 37.2-38.9 img/s against NVMe 33.9-37.3 on identical files | #40 comment 2026-10-28 |
| OpenSearch write ceiling | 8 writers 4,426 docs/s at 100,000 real vectors (about 740 needed at the target) | [PERFORMANCE "Stage 4"](../../PERFORMANCE.md#stage-4-coalesced-ingest-writes-lucene-engine-write-ceiling-on-the-final-mapping-248-254-230) |
| DALI ensemble, 20 MP JPEG in, YOLO26n 640 FP16 (img/s, p99 ms) | DALI GPU decode 14.6 (93) / 44.5 (264) / 106.2 (396) / 119.2 (1537) at 1 / 8 / 32 / 128 clients; DALI CPU decode caps at 27.6; client decode 97.6 at 128; an independent nvJPEG + GPU letterbox server 178.0 at 128 (pipelining, not the codec) | [dali_gpu_decode_lessons.md](../../research/dali_gpu_decode_lessons.md) 5 |
| DALI letterbox parity traps | `fn.resize` defaults `antialias=True`, `subpixel_scale=True`; Ultralytics pad split `round(d - 0.1)`; GPU shapes unreadable in the Triton DALI backend (use `fn.peek_image_shape`); after the fixes best shift (0,0), mean abs diff 0.08 grey | same, 3 |
| Port-mapped Docker networking | raw-tensor TRT FP16 112 to 145 img/s at 8 clients, 219 to 260 at 32 with host networking | same, 2 |
| Output payload | a `[batch, 100800, 85]` FP32 output makes batch 8 slower than batch 1 (3.60 / 3.23 / 2.07 img/s); the end2end detector scales 18.67 / 59.62 / 63.44 | [triton_perf_crossvalidation.md](../../research/triton_perf_crossvalidation.md) |
| Host | 2 x Xeon E5-2680 v3 (24 physical cores, 48 threads, 2 NUMA nodes), PCIe 3.0 x16; GPU 0 and 3080 Ti on node 0, GPU 2 on node 1, GPU 0 to GPU 2 is `SYS`; 300 W per A6000; driver 615.71.10, CUDA 13.4 | trade study 1.2; log SP0 |
| Device numbering | CUDA numbers cards fastest first, unlike `nvidia-smi` | [PERFORMANCE "Finding: CUDA numbers cards fastest first"](../../PERFORMANCE.md#finding-cuda-numbers-cards-fastest-first) |

### 2.6 What the numbers say, per workload

| Workload | Binding resource now | Evidence | What can still move it |
|---|---|---|---|
| Small images, policy `all` | PE tensor compute | encoder 95-97 % busy; 99.2 % of FLOPs; 57 % of FP16 peak | second card (1.9x measured); opt-in INT8 (1.2-1.4x if a recipe passes); fewer PE calls (a product rule, not a speed change) |
| Photos, ingest profile | the GPU, by `nvidia-smi` (94-97 %) | 48.3 img/s; PE about 15 ms of the 20.8 ms per photo (3.4 embeddings at 4.4 ms) | the non-PE GPU time per photo (decode on the SMs, kernels, detector, copies: about 6 ms, an estimate to be measured in PR-1/PR-3); the second card; whether "94 % busy" hides idle SMs (PR-0 records SM active) |
| Photos, default load list | device memory | 36.5 GB of resident models leaves room for one instance | load profiles, smaller per-instance memory (one process with many streams) |
| Small images, detect only and stateless routes | host CPU and per-request Python | `/detect` 133 img/s with the detector at 1.4 ms; the GPU route lost 21-49 % on frames | batched GPU decode with a dynamic batcher in front (PR-1, PR-4) |
| Faces, `/analyze` | host CPU (FP32 tensors, Python anchor decode, host alignment) | 0.36-0.79 API CPU-s per photo against 1.8-3.1 GPU ms | uint8 inputs, in-graph decode and NMS, batched GPU alignment (PR-5) |
| OCR | one Python model instance, sequential recognition | 45-48 ms of BLS per image for about 9 ms of GPU | concurrent recognition (5c-O1), argmax in the graph (PR-6) |

Roofline ceilings per A6000 (estimates; source: the earlier version of this plan, section 1.3, from
the measured costs above): PE at 57 % of peak about 31 COCO img/s and about 88 photos/s at 2.6
embeddings per photo (set B: 1.6 detections + frame); host CPU about 70-140 photos/s at 0.34
CPU-s per photo; PCIe over 1,000 photos/s; OpenSearch about 600 COCO img/s equivalent. With set P's
2.41 crops per photo (log SP0.1) the PE ceiling is about 67 photos/s.

## 3. Literature and industry review

### 3.1 Papers

| Source | Finding that applies here |
|---|---|
| "Beyond Inference: Performance Analysis of DNN Server Overheads for Computer Vision", arXiv [2403.12981](https://arxiv.org/abs/2403.12981) (v1 titled "A Deep Learning Accelerator is NOT All You Need") | on Triton, preprocessing can exceed 50 % of zero-load latency even with GPU preprocessing; with large images (9.5 MB, 3564 x 2880) end-to-end throughput fell to 19.5 % of what the inference stage could reach, and more GPUs did not always help. Same shape as our photos before the GPU decode |
| Kang et al., "Jointly Optimizing Preprocessing and Inference for DNN-based Visual Analytics" (Smol), PVLDB 14 (2021), [pdf](http://www.vldb.org/pvldb/vol14/p87-kang.pdf) | preprocessing (decode, transform, transfer) is the end-to-end bottleneck once the model is fast; reduced-resolution (partial and thumbnail) JPEG decode is a first-class optimization when accuracy allows. Supports 5c-K4 and PR-1's reduced-decode column |
| Weißenberger and Schmidt, "Accelerating JPEG Decompression on GPUs", arXiv [2111.09219](https://arxiv.org/abs/2111.09219) | GPU Huffman decoding by self-synchronization makes the entropy stage parallel; the technique behind GPU Huffman backends. Explains why GPU Huffman costs SM time that a PE-bound card does not have spare |
| Crankshaw et al., "Clipper", NSDI 2017, [paper](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw) | adaptive batching under a latency objective; the same principle as Triton's dynamic batcher with a queue delay, and the reason per-image workers feeding the models' own batchers beat a barrier batch (section 2.2) |
| Shen et al., "Nexus: a GPU Cluster Engine for Accelerating DNN-Based Video Analysis", SOSP 2019, [doi](https://doi.org/10.1145/3341301.3359658) | batch-aware placement and splitting a request's models across GPUs by measured per-batch cost; supports one replica per card with routing instead of spreading one image across cards |
| Xiao et al., "SmoothQuant", ICML 2023, arXiv [2211.10438](https://arxiv.org/abs/2211.10438) | migrating activation outliers into weights makes W8A8 work for transformers; the recipe for the opt-in INT8 round (PR-9) |

### 3.2 NVIDIA libraries and server features (official docs, read 2026-11-04)

| Library or feature | What the docs say that matters | Source |
|---|---|---|
| nvJPEG 13.5 | backends: `HYBRID` (Huffman on the CPU), `GPU_HYBRID` (Huffman on the GPU), `HARDWARE` (hardware decoder), `GPU_HYBRID_DEVICE` (device-memory input, batched only, baseline without restart intervals); `nvjpegDecodeBatched` uses GPU Huffman for baseline interleaved JPEGs only when the batch is **greater than 50**; the hardware decoder exists on A100, A30, Hopper, Ada, Blackwell, Jetson Thor (not GA102); scaled output (1/2, 1/4, 1/8) and batched ROI decode **only with the `HARDWARE` backend**; `NVJPEG_FLAGS_REDUCED_MEMORY_DECODE` for `HYBRID` and `GPU_HYBRID`; known limit: decoupled APIs with `GPU_HYBRID` may misdecode out-of-bound run-length codes | [nvJPEG docs](https://docs.nvidia.com/cuda/nvjpeg/index.html), sections "Backend", "nvjpegDecodeBatched", "ROI decode", "Known issues" |
| nvImageCodec (release notes up to 0.10.0-beta) | `Backend(backend_kind, load_hint, load_hint_policy)` with `GPU_ONLY`, `HYBRID_CPU_GPU`, `HW_GPU_ONLY`, `CPU_ONLY`; `DecodeParams(apply_exif_orientation=True, allow_any_depth, color_spec)`; decoder `options` string (`:fancy_upsampling=1`, `:num_cuda_streams=4`); `decode(..., cuda_stream=)`; batch decode returns `None` for failed positions; ROI through `Region` and `CodeStream.get_sub_code_stream`; `Encoder` (`nvjpeg_cuda_encoder`), which raises for chroma-subsampled image objects; zero-copy through `__cuda_array_interface__` to CuPy and CV-CUDA | [Python API](https://docs.nvidia.com/cuda/nvimagecodec/py_api.html); [overview](https://docs.nvidia.com/cuda/nvimagecodec/) |
| DALI 2.x `fn.decoders.image` | `hybrid_huffman_threshold` default 1,000,000 pixels (larger images use the nvJPEG hybrid Huffman decoder); `adjust_orientation` default **True**; `jpeg_fancy_upsampling` default **False** (must be True to match the CPU decoder's chroma); `hw_decoder_load` 0.65 (only where a hardware decoder exists); `preallocate_width_hint` / `height_hint` for the hardware decoder | [decoders.image](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html) |
| DALI dynamic mode (2.0+) | imperative `nvidia.dali.experimental.dynamic` (`ndd`), asynchronous and deferred execution; eager and debug modes removed in 2.0; capture mode in 2.3; experimental | [DALI 2.0 release notes](https://docs.nvidia.com/deeplearning/dali/archives/dali_2_0_0/release-notes/index.html), [dynamic API overview](https://docs.nvidia.com/deeplearning/dali/archives/dali_2_0_0/user-guide/dali_dynamic/overview.html) |
| Triton DALI backend | autoserialized pipelines (`@autoserialize`); the decoder accepts CPU input; the README says Triton accepts only homogeneous batch shapes, so differently sized encoded images need padding (our 2026-09 run used `allow_ragged_batch: true`; re-check on the pinned release); "We suggest using default instance group for DALI model" (more instances raise memory) | [dali_backend README](https://github.com/triton-inference-server/dali_backend) |
| CV-CUDA 0.19.0 (2026-10) | operators `resize` (with `Interp.AREA`), `pillowresize`, `resize_crop_convert_reformat`, `copymakeborder`, `warp_affine`, `warp_perspective`, `laplacian`, `nms`, `normalize`, `convertto`; `ImageBatchVarShape`; explicit `Stream`; interop with PyTorch, CuPy and nvImageCodec; 0.19 adds a torchvision-compatible `cvcuda.tv` API and fixes stream-handle lifetime | [CV-CUDA docs](https://cvcuda.github.io/CV-CUDA/); [v0.19.0 release](https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.19.0) |
| Triton Python backend | one process per model instance because of the GIL; BLS `async_exec` needs `async def execute`; "Async BLS is not supported when running a Python model in decoupled mode"; `async def execute` is supported for decoupled models from 24.04; DLPack zero-copy (`to_dlpack`, `from_dlpack`); `preferred_memory` on `InferenceRequest` (23.04+); `FORCE_CPU_ONLY_INPUT_TENSORS` "no" keeps inputs on the GPU; conda-pack custom environments | [python_backend README](https://github.com/triton-inference-server/python_backend), sections "Multiple Model Instance Support", "Business Logic Scripting", "Interoperability and GPU Support", "Input Tensor Device Placement" |
| Triton ensembles | a fixed dataflow, no conditional branches; no `instance_group` on the ensemble; `max_inflight_requests` bounds intermediate requests (otherwise fast upstream steps can run the card out of memory, "especially common with decoupled models") | [ensemble models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html) |
| Triton model configuration and optimization | rate limiter `resources` and `priority` per instance group; warmup blocks READY; several instance groups per model with `gpus` lists; "the dynamic batcher alone is capable of fully utilizing the GPU and so adding additional model instances does not provide any performance advantage" (typical case); concurrency `2 * max batch * instances` for maximum throughput; host policy restricts memory allocation to a NUMA node | [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md); [optimization, "NUMA Optimization"](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) |
| Triton dynamic batcher, CUDA graphs | batches only same-shape inputs unless ragged batching is set up; `priority_levels`; CUDA graphs in `optimization { cuda { graphs, graph_spec } }`, "Currently only recognized by TensorRT backend" | [batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html); [model_config.proto](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) |
| Triton C++ backend API | a native backend gets requests through `TRITONBACKEND_*`, can issue in-process inference through the server C API (the Python backend's BLS does this) | [backend repo](https://github.com/triton-inference-server/backend) |
| Triton CUDA shared memory | clients on the same host register CUDA IPC regions for inputs and outputs | [shared-memory extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html) |
| TensorRT 11 | strongly typed networks: precision comes from the ONNX (FP16 baked by ModelOpt AutoCast in this repo), not builder flags; engines bound to TensorRT version and GPU architecture | [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/); `Dockerfile.triton` lockstep check |
| CUDA MPS | the control daemon and server let several processes share one GPU co-operatively, with memory and SM partitioning; a daemon outside the containers | [MPS docs](https://docs.nvidia.com/deploy/mps/index.html) |
| DCGM profiling metrics | `DCGM_FI_PROF_SM_ACTIVE` (cycles with at least one warp per SM), `DCGM_FI_PROF_SM_OCCUPANCY`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (tensor pipe active cycles); `nvidia-smi` utilization only says a kernel ran in the sample window | [DCGM feature overview, profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics) |
| Nsight Systems | CUDA API trace (launches, syncs, mallocs, copies, gaps) per process | [user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) |
| NVIDIA Model Optimizer | `mtq.quantize` with `INT8_SMOOTHQUANT_CFG`; `mtq.auto_quantize` per-layer format search with FP16 allowed | [quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html) |
| torchvision `decode_jpeg` | batched GPU decode from a list of CPU tensors; `apply_exif_orientation` CPU only; image decoding deprecated since torchvision 0.29 in favour of TorchCodec | [decode_jpeg](https://docs.pytorch.org/vision/main/generated/torchvision.io.decode_jpeg.html) |

### 3.3 Mature open-source references

| Reference | Pattern | What we take, what we do not |
|---|---|---|
| Immich machine-learning service | one model server process by default (`MACHINE_LEARNING_WORKERS` 1, "each process duplicates the models"), a request thread pool (`MACHINE_LEARNING_REQUEST_THREADS`), per-model batch caps (`MACHINE_LEARNING_MAX_BATCH_SIZE__FACIAL_RECOGNITION` 4, `..._OCR` 6), models unloaded after `MACHINE_LEARNING_MODEL_TTL` (300 s), preloading per model, multi-GPU by one worker per device round-robin (`MACHINE_LEARNING_DEVICE_IDS`) ([env vars](https://docs.immich.app/install/environment-variables), [service](https://github.com/immich-app/immich/tree/main/machine-learning)) | take: per-deployment residency and per-model batch caps as configuration (our load profiles and serving profile, seam S-5), one worker per device for multi-GPU. Do not take: models duplicated per process; one process per card is our unit instead |
| NVIDIA DeepStream (`nvstreammux`, `Gst-nvdspreprocess`, secondary inference) | decode once into batched GPU surfaces; ROIs of detected objects "scaled and format converted as per the network requirements" into one tensor for the secondary model (SGIE mode), passed as tensor metadata to `nvinfer` (`input-tensor-meta=1`) ([nvdspreprocess](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html)) | take: the exact dataflow we want (frame resident, crops cut at model size per consumer, batched per model). Do not take: the GStreamer runtime; it targets streams of video frames, not request-driven still images with per-request plans |
| FiftyOne `apply_model` | a torch `DataLoader` with worker processes decodes and batches; the model runs in the main process | a reminder that CPU-side decode with many processes is a valid baseline (our CPU route for small images); not a serving design |
| Triton DALI ensembles (the backend's examples) | JPEG in, DALI decode and resize, model, in one ensemble | candidate for the stateless routes (section 4, D-routes), not for the conditional per-detection fan-out |

### 3.4 What the review confirms

- The serving layer cannot speed up PE's kernels; only the second card, precision and fewer PE
  calls move small-image throughput (trade study 0; PE pass: 57 % of peak, fused attention already
  selected).
- Decode-once with model-size views per consumer is the industry pattern (DeepStream), and the
  repo already has it for photos (`gpu_image_prep`). The open questions are cost per photo, memory
  per instance and the host of the executor, not the pattern.
- Per-image work feeding each model's own dynamic batcher is the documented way to batch across
  requests (Triton batcher docs; Clipper); the repo measured the barrier alternative at 15 photos/s
  against 43-48.

### 3.5 What the review changes in the repo's assumptions

| Old assumption | Where | Correction (source) | Consequence |
|---|---|---|---|
| a reduced (DCT-scaled) decode could feed the GPU route (5c-K4 "read the nvImageCodec decode parameters") | [06c](06c_b5_crops_ocr_faces.md) 5c-K4 | nvJPEG scaled decode needs the `HARDWARE` backend (nvJPEG docs); the A6000 has none | 5c-K4 is a CPU-route measurement only (Pillow `draft`, OpenCV `IMREAD_REDUCED_*`) |
| nvImageCodec's GPU backend does not decode on this card, so a custom ctypes path is needed | `nvjpeg_lanes.py` docstring | nvJPEG's batched API uses GPU Huffman only above a batch of 50; per-image `decode` calls never reach it. nvImageCodec 0.7's `GPU_ONLY` result may be this rule, not a missing feature | PR-1 tests nvImageCodec's current release at batch 64 before keeping the custom lanes (J8) |
| DALI matches the CPU decoder if EXIF is handled | [06b](06b_b5_decode_once.md) 5b-G3 | `jpeg_fancy_upsampling` defaults to False (chroma differs from libjpeg); `adjust_orientation` defaults to True | 5b-G3 and PR-1 set both explicitly and test them |
| DALI handles ragged JPEG batches in Triton | [dali_gpu_decode_lessons.md](../../research/dali_gpu_decode_lessons.md) 4 | the backend README says only homogeneous batch shapes | verify on the pinned Triton before 5b-W1 |
| "GPU busy 94-97 %" means the card is full | log SPF.4, #407 | `nvidia-smi` utilization only says a kernel ran (DCGM docs) | PR-0 records SM active and tensor active; #407's ceiling is re-read in those units |
| profile the image-prep model with Nsight Systems inside the Triton container | [06b](06b_b5_decode_once.md) 5b-P1 | `Dockerfile.triton` deletes the Nsight Systems CLI (`rm -rf /usr/local/cuda-*/NsightSystems-cli-*`) | 5b-P1 mounts the host's `nsys` read-only into its bench stack, or profiles the venv prototype (PR-3) |
| torchvision GPU decode as an arm | earlier decoder matrix | deprecated since torchvision 0.29 (TorchCodec supersedes), no EXIF on CUDA | dropped from every matrix |
| the GIL limits the image-prep model | #407 stage 3, the earlier H-1 | no saturated thread at the measured load (log SP0.7.5) | the C++ backend opens only on the PR-4 gap bar (section 4.3) |

## 4. Candidate designs

Legend: ✅ recommended, ❌ rejected (reason given), 🟡 later (named evidence reopens it).

### 4.1 Where decode and the resident frame live (the executor host)

| ID | Design | Verdict | Reason |
|---|---|---|---|
| X-a | Inside Triton, next to the engines, one executor model that both the API (interactive) and the Celery workers (bulk) call | ✅ | one preprocessing implementation for both paths; the frame never leaves the card of its consumers; BLS calls are in-server; the shipped design measured 48.3 photos/s with 0 fallbacks |
| X-b | Executor as a Triton Python model (today's `gpu_image_prep`), its stages backed by libraries chosen in PR-1/PR-2 | ✅ default | the GIL was not saturated (SP0.7.5); per-image workers reach the batcher; library ops replace custom kernels only on parity |
| X-c | Executor as a native C++ Triton backend | 🟡 | opens only if PR-4 shows the Python executor costs at least 25 % against the same pipeline run directly (PR-3) at the bulk point while SM active is under 85 % (bar from the earlier E-3/E-7); then a plan file of its own (OQ-2 of the earlier version, J6 here) |
| X-d | Decode and preprocessing in a Celery GPU worker per card, sending tensors to Triton through CUDA shared memory | ❌ for 0.7.0, 🟡 after | two places that own GPU memory on one card (Triton and the worker), CUDA IPC across processes, and a second path for the interactive routes (a Celery hop is not an RPC); reopens only if X-b and X-c both miss and PR-3's direct number is much higher than PR-4's |
| X-e | Run TensorRT engines directly in the bulk workers (no Triton for bulk) | ❌ | two serving paths for the same engines and two memory owners per card; architecture E of the trade study, rejected unless the bulk runner misses by more than 15 % (trade study 7) |
| X-f | DeepStream pipeline | ❌ | built for video streams in GStreamer; no request-driven plans or per-request conditions; the pattern is taken (3.3), the runtime is not |
| X-g | Share a decoded frame across processes (VLM, segmenter, region workers) through CUDA IPC | ❌ | lifetime and memory across processes and replicas; the multi-user rule forbids process-local shared state; work crops at model size (#219) remove the need. The earlier H-12, closed |

### 4.2 Decoder

| ID | Design | Verdict | Reason |
|---|---|---|---|
| D-a | nvJPEG `GPU_HYBRID` decoupled API per image (today's ctypes lanes) | ✅ baseline | measured, EXIF and chroma handled, 0 fallbacks on the ingest profile; custom code justified because no library reached GPU Huffman per image on this card (section 2.1) |
| D-b | nvImageCodec current release, batched `GPU_ONLY` at batch over 50, `HYBRID_CPU_GPU` otherwise | 🟡 decided by PR-1 | replaces the ctypes lanes only if it reaches GPU Huffman at real batch sizes with equal pixels; batches over 50 need requests to meet, which per-image workers do not do today |
| D-c | DALI `mixed` decoder (`hybrid_huffman_threshold`, `jpeg_fancy_upsampling=True`) inside a DALI model or dynamic mode | 🟡 decided by PR-1, 5b-G3 | measured 31.8 photos/s and 3.8 GPU ms per photo with letterbox (`dali_full`); worth it only for decode + letterbox where a fixed graph fits (stateless routes) |
| D-d | CPU Huffman (hybrid) on a GPU-bound card | 🟡 decided by PR-1 | trades about 7.5 CPU-ms per MP for SM time; at 48 photos/s of 21 MP that is about 7.5 cores, on a host shared with OpenSearch; only if PR-1 shows GPU Huffman costs the card more than 3 ms per photo and the host has the cores |
| D-e | Reduced (DCT-scaled) decode on the GPU route | ❌ | needs nvJPEG's `HARDWARE` backend, absent on GA102 (nvJPEG docs); the CPU-route variant stays 5c-K4 |
| D-f | torchvision GPU decode | ❌ | deprecated (0.29), no EXIF on CUDA, serial at 4.5 photos/s |
| D-g | GPU decode for progressive JPEGs | 🟡 | 11 % of set B; nvJPEG GPU backends cover baseline only; measure the CPU share they cost after D-a/D-b (5b-C1 b) |
| D-h | GPU decode for small images (COCO size) on the bulk path | ❌ | the GPU route lost 21-49 % on frames; PE binds there; the 3 MP gate stays. Stateless routes are D-routes below |

### 4.3 Resize, crop, color and blur operations

| ID | Design | Verdict | Reason |
|---|---|---|---|
| K-a | Today's fused uint8 CuPy kernels (Pillow-`BILINEAR` letterbox and frame, plain bilinear squares, exact blur sums) | ✅ baseline | 1.3 ms GPU per 18 MP photo, byte-identical to the CPU route's definition |
| K-b | CV-CUDA `pillowresize`, `resize`, `copymakeborder`, `laplacian`, `resize_crop_convert_reformat` on `ImageBatchVarShape` | 🟡 decided by PR-2 / 5b-G2 | adopt if byte-identical, or within the tensor gate and the task gates (07b 5b-G2 rule); a maintained library replaces 198 lines of custom kernels only on parity |
| K-c | DALI `fn.resize` letterbox | 🟡 | only inside D-c; needs the three documented parity settings |
| K-d | Cascade crops or the frame square from the 640 letterbox | ❌ | measured destructive: crops p01 0.534 on photos, frame 0.945 (gpu_decode_plan 6.5) |
| K-e | Work crops at model size (`crop_max_side`, VLM cap) | ✅ (#219, 5c) | bytes and host work; the GPU route applies the same cap before encode |
| K-f | GPU JPEG encode of work crops (nvImageCodec `Encoder`) | 🟡 decided by 5b-C1 | 12 % of stage time at one batch, 4.8-5.1 % at 4 batches; small once crops are capped |
| K-g | Face alignment with CV-CUDA `warp_affine` on the resident frame | 🟡 decided by PR-5 / 5d-P5 | replaces host `cv2.warpAffine`; gate ArcFace p01 >= 0.995, P-face 1.0 |

### 4.4 Orchestration inside the server

| ID | Design | Verdict | Reason |
|---|---|---|---|
| O-a | Per-image worker threads in one decoupled Python model, BLS calls into the engines' dynamic batchers, bounded in flight and by a byte budget | ✅ | measured best (43-48 against 15 for a barrier batch); the budget turned silent fallbacks into counted waits |
| O-b | Triton ensembles for the conditional plan | ❌ | fixed dataflow, no conditions, no per-detection fan-out (ensemble docs; `processing_plans_plan.md` 5.3) |
| O-c | DALI ensemble for stateless detect-only and embed-image routes | 🟡 decided by 5b-W1 | a fixed graph fits there; bar 1.5x the CPU arm at 32 clients |
| O-d | Async BLS in the image-prep model | ❌ while it is decoupled | "Async BLS is not supported ... in decoupled mode" (python_backend README); re-read on the pinned release (PR-4 records it) |
| O-e | Non-decoupled OCR model with `async_exec` per crop | ✅ (5c-O1) | qualifies (not decoupled); in-flight overlap, not cross-width batching |

### 4.5 Placement across cards, memory, scheduling, precision

| ID | Design | Verdict | Reason |
|---|---|---|---|
| M-a | One Triton server per A6000, NUMA-pinned to its node, each holding the whole load list; a client router picks the server (least outstanding requests) | ✅ target, decided by PR-7 / E-11 | the resident frame must sit beside its consumers; GPU 0 and GPU 2 are `SYS` apart; data-parallel PE reached 1.91x |
| M-b | One Triton with `instance_group { gpus: [0, 2] }` for every model | 🟡 the comparison arm of PR-7 | simpler to run; the executor's BLS calls may land on the other card's engine instance, crossing the socket with the frame |
| M-c | The 3080 Ti (GPU 1) | ❌ by etiquette | belongs to other workloads; only #42's acceptance with consent |
| M-d | Load profiles that keep only the models a deployment uses resident (the ingest load profile) | ✅ (shipped) | freed 27.7 GB; becomes part of the serving profile (seam S-5) |
| M-e | CUDA MPS | 🟡 parked (#403) | the 2-instance gap was memory; reopens only if PR-3/PR-4 show two contexts time-slicing (idle gaps while both are active); opt-in overlay, host root |
| S-a | Triton `priority_levels` for interactive over bulk on the same server | ✅ decided by 5a-S6 and E-12 | the documented mechanism; interactive p99 <= 2x idle with bulk >= 95 % of solo |
| Q-a | FP16 everywhere | ✅ | the only precision that passed the gates |
| Q-b | INT8 PE, second round (SmoothQuant, per-layer search) | 🟡 opt-in only (PR-9) | 1.15-1.33x if every gate passes; never a default |
| Q-c | CUDA graphs on the small fixed-shape engines at batch 1 | 🟡 (5a-S5, E-5) | PE and detector lost about 2 %; small engines may gain at batch 1 |
| Q-d | 2:4 sparsity, FP8 | ❌ | SM86 has no FP8; sparsity changes the model (trade study 9) |

### 4.6 Transport between the API or workers and Triton

| ID | Design | Verdict | Reason |
|---|---|---|---|
| T-a | JPEG bytes over gRPC for photos; uint8 tensors for small images | ✅ | 10.4 MB per photo instead of 54 MB; uint8 cut 4x |
| T-b | System or CUDA shared memory from the API | ❌ | under 5 % of API CPU; the API owns no GPU (B4); CUDA shared memory from multithreaded clients is a known issue (trade study 3.1) |
| T-c | Host networking for the bench stacks | ✅ for measurements | Docker's userland proxy cost 23 % at 8 clients (lessons 2); production keeps the compose network |

## 5. Seams the v0.7.0 rebuild must include

Each seam is something the database and queue rebuild must provide now so that the winning design
lands later with no schema, queue or service-shape change. Each is small, and each has a red test.
"Owner" is the plan file and lane that should carry it; section 11 repeats them as edit
instructions for the integration agent.

### 5.1 Schema (first Alembic revision `0001_control_plane`)

| ID | Seam | Exact shape | Why | Owner |
|---|---|---|---|---|
| S-1 | Engine input contract on the model registry | `model_engines.input_contract jsonb NOT NULL`, validated on write and read by a pydantic model `EngineInputContract` in `src/schemas/models.py`: `dtype` (`uint8`/`fp16`/`fp32`), `layout` (`NCHW`), `size` (`[h, w]` or dynamic ranges `{min, opt, max}`), `color` (`RGB`/`BGR`), `normalize` (`in_graph` or `{mean, std, scale}`), `resize` (`letterbox`/`short_edge_center_crop`/`stretch`/`native_width`), `interpolation` (`pil_bilinear`/`bilinear`/`area`), `fill` (int), `pad_split` (`floor_half`/`ultralytics_round`), `max_side` (int, optional) | one definition of every model input for the CPU route, the GPU route and the prototypes; a decoder or kernel change becomes a contract diff with a gate, not a hidden edit in two files (GPU-12 was exactly such a drift) | [02a](02a_postgres_schema.md) section 7; schema lane 1A ([03a](03a_stage1_foundation.md)); written by lane 5D/5B ([03e](03e_stage5_gpu_queues.md)) |
| S-2 | Engine output contract | `model_engines.output_contract jsonb NOT NULL` (`EngineOutputContract`: `kind` `end2end_boxes`/`embedding`/`anchors_raw`/`ctc_logits`/`ctc_argmax`, shapes, dtypes) | in-graph post-processing (SCRFD decode and NMS, OCR argmax, PR-5/PR-6) changes outputs; consumers check the contract instead of the model name | same |
| S-3 | Build profile, settings and stamp | `model_engines.build_profile text NOT NULL DEFAULT 'default' CHECK (build_profile IN ('default','small'))`; `build_settings jsonb NOT NULL` (workspace MiB, opt/max batch, builder level, precision recipe); `build_stamp text NOT NULL`; `onnx_sha256 text`, `engine_sha256 text`; the unique key becomes `UNIQUE (triton_model_name, trt_version, gpu_arch, build_profile)` | #42's small target and #415's stamps; an INT8 opt-in build (PR-9) is a different row, not an overwrite | same |
| S-4 | Measured serving cost per engine | `model_engines.device_mib_measured integer NULL`, `max_batch integer NOT NULL`, `opt_batch integer NULL`, `cuda_graphs boolean NOT NULL DEFAULT false`, `precision text NOT NULL CHECK (precision IN ('fp32','fp16','int8'))`, `parity_ref text NULL` (the capture tag that approved the engine), `parity_report_key text NULL` (storage key) | the serving planner (load profiles, instances per card) reads measured memory instead of constants (`8704` today is a hand-copied number); precision is a fact of the engine, and INT8 is visible | same |
| S-5 | Deployment serving profile | new config kind `serving_profile` in `config_documents` (`project_id` NULL); the CHECK becomes `CHECK ((kind IN ('vlm_endpoint','serving_profile')) = (project_id IS NULL))`; body `ServingProfile`: `servers` (one per card: name, gRPC and metrics URLs, GPU UUIDs, NUMA node), `load_list` per server, `instances` per model, `image_prep` (`instances`, `workers`, `max_in_flight`, `device_budget_mib`, `min_megapixels`), `priorities` (interactive, bulk) | today these live in `config_templates/profiles/*.json`, `.env` and the installer's edits of `config.pbtxt`; revisioned and auditable, one place for the router and for `openprocessor gpu plan` | [02a](02a_postgres_schema.md) section 4; lane 2C ([03b](03b_stage2_control_data.md)) |
| S-6 | GPU inventory by UUID | new table `gpu_devices (gpu_uuid text PRIMARY KEY, host text NOT NULL, nvml_index integer NOT NULL, cuda_index integer NULL, name text NOT NULL, compute_capability text NOT NULL, memory_total_mib integer NOT NULL, numa_node integer NULL, pcie_gen integer NULL, pcie_width integer NULL, hw_jpeg_decoder boolean NOT NULL, driver_version text NOT NULL, cuda_version text NOT NULL, seen_at timestamptz NOT NULL DEFAULT now())`, deployment-scoped (no RLS), retention as `worker_capabilities` | CUDA numbers cards fastest first (PERFORMANCE finding), so an index is ambiguous across processes; `hw_jpeg_decoder` decides D-b/D-c defaults on other cards (A100 class) without code changes | [02a](02a_postgres_schema.md) section 5; lane 1A; writer: each GPU-holding worker at `worker_ready` through NVML, plus `openprocessor gpu plan` |
| S-7 | GPU reservations keyed by UUID | `gpu_reservations` primary key `(host, gpu_index)` becomes `(gpu_uuid)` referencing `gpu_devices` | same ordering problem; the arbiter and training reserve by identity | [02a](02a_postgres_schema.md) section 5; lane 1A; users in 03c 6A and 03f 5A |
| S-8 | Per-job performance summary | `jobs.result` for `bulk_ingest_run`, `reprocess`, `directory_ingest` carries a validated `JobPerfSummary`: images, wall s, images/s, decode routes (`gpu`, `cpu_small`, `cpu_progressive`, `cpu_gpu_memory`, ...), stage seconds per stage, PE calls per image, server names used | the scale ladders (#45) and the R-final rows read throughput from the job record instead of a side script; no new column | [02b](02b_jobs_locking_events.md) section 2; lane 3A writes the model, lanes 4C and 4A fill it ([03d](03d_stage4_job_families.md)) |
| S-9 | Resolved processing plan per job | `jobs.params.plan` (the resolved `ProcessingPlan` from `src/domain/processing_plan.py`) and `jobs.params.plan_sha256` | a job's branches are auditable and comparable across runs; 5d-P4 narrowing needs it | [02b](02b_jobs_locking_events.md) section 2.2; lane 4C |
| S-10 | Work-crop variants in the storage seam | crop cache keys include the size variant: `projects/<slug>/crops/<item>/<box>/<max_side>.jpg` (`0` = source size) | #219 changes crop size; a cached source-size crop must never be served as a capped one | [02c](02c_migration_ops_tests.md) section 5; lane 1E ([03a](03a_stage1_foundation.md)) |

No new table for benchmark results: raw runs stay in `artifacts_local/` and published rows in
`docs/PERFORMANCE.md` (07 M-8, M-9).

### 5.2 Queues, job kinds and the GPU worker shape

| ID | Seam | Exact shape | Owner |
|---|---|---|---|
| S-11 | No GPU-prep queue in 0.7.0, by design | bulk ingest and reprocess stay on `jobs` workers that hold no GPU and call the executor in Triton (X-a); the kind table gains a field `gpu_affinity` with values `none` (default), `any_card`; the kinds that call the executor set `any_card`, and the job wrapper passes the router's chosen server name into the handler and records it in `JobPerfSummary.servers` | [02b](02b_jobs_locking_events.md) section 2.1; lane 3A |
| S-12 | GPU worker shape reserved for later device-scoped queues | `src/tasks/runtime/gpu_app.py::make_app` takes `device_uuid` and derives queue names `<family>.<slot>` from `gpu_devices` when a family is declared device-scoped (none is in 0.7.0); CUDA initialized only in the child after fork (`worker_process_init`), never in the parent (CUDA does not support fork after initialization; `tests/tasks/test_no_cuda_import.py` already guards the parent); `--pool prefork -c 1` per card | [02b](02b_jobs_locking_events.md) section 7.3; [03e](03e_stage5_gpu_queues.md) |
| S-13 | Export job records the engine facts | `model_export` (on `gpu_export`) writes S-1..S-4 for every engine it builds, measures `device_mib_measured` by loading the engine once, and refuses (fails the job) when the ONNX input does not match the declared contract | [03e](03e_stage5_gpu_queues.md) lane 5B |
| S-14 | Serving inventory refresh | a beat task `op.periodic.serving_inventory` (periodic queue, 60 s) reads each server of the serving profile (`/v2/repository/index`, model readiness, metrics) and writes `config_runtime_reports`-style rows `serving_reports (server text, model text, version integer, state text, instances integer, device_mib integer, seen_at timestamptz, PRIMARY KEY (server, model, version))` | [02a](02a_postgres_schema.md) section 4; [03f](03f_stage6_scheduler_scaleout.md) 6A |

### 5.3 Clients, compose and configuration

| ID | Seam | Exact shape | Owner |
|---|---|---|---|
| S-15 | Triton router | `src/clients/triton_client_manager.py` takes its server list from the active serving profile (one server by default) and picks per call the server with the fewest outstanding requests in this process among those whose `serving_reports` say the model is ready; the image-prep call and every follow-up for that image go to the same server | [05](05_b4_stateless_api.md) (lane 4B, pools from totals) |
| S-16 | Triton as a per-card service in compose | the `triton-server` service has no fixed `container_name` (D-54 already) and reads its card and NUMA node from the serving profile's generated env (`TRITON_GPU_ID` today); a second card is a second service instance (`triton-server-<slot>`) generated by the installer, not a hand edit | [03f](03f_stage6_scheduler_scaleout.md) 6B |
| S-17 | One geometry module shared by both routes | the letterbox, square, crop-rectangle and un-map functions live in one pure module (`src/domain/image_geometry.py`), and `export/sync_model_seed.py` copies it into the image-prep model directory at seed time; a test asserts the copy is byte-identical. Replaces today's pair `src/services/detection/geometry.py` and `models/gpu_image_prep/1/prep_geometry.py`, which are held equal only by tests | B5b lane `prep` ([06b](06b_b5_decode_once.md) 5b-G1) |

### 5.4 Metrics

| ID | Metric | Type, labels | Owner |
|---|---|---|---|
| S-18 | `op_decode_route_total` | counter `{route, path}` (`path` = `ingest`/`reprocess`/`analyze`/`detect`/...): one name for today's `op_ingest_decode_total` and `op_reprocess_decode_total` | 5b-D1 (renamed with its helper, no alias) |
| S-19 | `op_gpu_prep_stage_seconds`, `op_gpu_prep_stage_observations` | keep; add label `server` | B5b |
| S-20 | `op_gpu_prep_budget_bytes_in_use`, `op_gpu_prep_frames_in_flight` | gauges `{server}` | B5b `prep` |
| S-21 | `op_plan_branch_total` | counter `{route, branch, outcome}` (`ran`, `not_run`, `error`) | 5d-P2 |
| S-22 | DCGM profiling fields in the monitoring profile | the DCGM exporter's field list adds `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_SM_OCCUPANCY`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_PROF_PCIE_TX_BYTES`, `DCGM_FI_PROF_PCIE_RX_BYTES`; the Grafana GPU dashboard shows them beside utilization | [03f](03f_stage6_scheduler_scaleout.md) 6B (monitoring) |
| S-23 | `op_triton_router_outstanding` | gauge `{server}` per process | B4 4B |

## 6. Prototype protocol

### 6.1 Where prototype code lives

**Recommendation: a tracked folder `scripts/research/gpu_pipeline/` with its own exact-pinned
requirements file, outputs only under the gitignored `artifacts_local/`.**

| Option | For | Against |
|---|---|---|
| Tracked `scripts/research/gpu_pipeline/` with `requirements-gpu-research.txt` (every line `==`) | `docs/research/CLAUDE.md` requires a reproduce command "that works from a clean checkout" for every result; reviewers can read the method; the folder has a precedent (`scripts/research/` with its own requirements files); reuses `scripts/bench/gpu_decode_lib.py`, `gpu_prep_parity.py`, `parity_gates.py` and the dataset fetchers; exact pins make a later re-run mean something | the ratchets (file size, complexity, broad excepts, unused code, naming scan) apply; scripts must take every path as an argument |
| Gitignored local folder | no review, no ratchets, fast | public numbers without reproducible code break the research rules; work is lost with the worktree; nothing to cite |

Rules for the folder: no import of `src.services` or `src.routers` (the prototypes may import
`src/domain/` and `scripts/bench/` helpers, and the image-prep model's modules
`models/gpu_image_prep/1/*.py` by path); every dataset path, output path and GPU index is an
argument; no default points at a private location; a README in the folder lists each prototype
with its command (written by the prototype lane, not by this plan). Results are files under
`artifacts_local/research/gpu_pipeline/<prototype>/<run_id>/`.

### 6.2 The venv

- Location: `artifacts_local/venvs/gpu-research/` (inside the gitignored `artifacts_local/`; the
  `.gitignore` rule `.venv` matches only that exact name, so a `.venv-*` sibling would not be
  ignored), separate from `.venv/` because it carries CUDA wheels the offline suite must not see
  (the CLAUDE.md note about `faiss-gpu-cu12` breaking `faiss-cpu` is the same hazard).
- Interpreter: the system Python 3.12, `python3 -m venv artifacts_local/venvs/gpu-research`;
  binaries called directly (`artifacts_local/venvs/gpu-research/bin/python`), never `source`.
- Pins, matched to the Triton image so engine plans and kernels compare (verify each on PyPI at
  lane start and write the exact version found): `tensorrt-cu13==11.1.0.106` (the
  `Dockerfile.triton` lockstep version, so the venv loads the same `.plan` files), `cupy-cuda13x==13.6.0`
  (the Triton image's pin), `nvidia-nvimgcodec-cu13` (newest), `nvidia-dali-cuda130` (newest 2.x),
  `cvcuda-cu13` (0.19.0 or newest), `numpy`, `pillow` and `opencv-python-headless==4.13.0.92` at the
  product pins, `simplejpeg`, `nvidia-ml-py`, `onnx`, `onnxruntime-gpu` at the product pin, `py-spy`.
  Host tools already present: `nsys`, `ncu` (CUDA toolkit), `numactl`/`taskset`.
- Engines: built in the venv with the repo's export scripts into `artifacts_local/research/engines/`
  (never into `models/`), or the engines of the running test stack read-only; `trtexec` from the
  venv's TensorRT wheel. A plan built elsewhere is used only if its `trt_version` and `gpu_arch`
  match (S-3's stamp).
- The through-Triton arm (PR-4) runs the stock pinned `tritonserver:26.06-py3` image by digest with
  the model repository mounted and host networking: pulled, never built; its own container name
  `op-research-triton-<user>`, never a compose project of the stack.

### 6.3 Datasets and sampling

| Set | Content | Size for prototypes | Use | Records |
|---|---|---|---|---|
| set B | 500 pinned Commons photos, 12-24 MP (`commons_hires_500.json`) | all 500; tier 1 the first 200 by manifest order | every photo row; the public number | public |
| set B-2k | the 2,000 extension (OQ-B5-1) if pinned | 2,000 | confirmation | public |
| set P | the owner's private photos, about 21 MP | a seeded sample of 1,000 photos (seed 20261104) copied, not linked, to a local NVMe scratch directory outside the repository; the copy list is a manifest of relative names and sha256 kept in the private ledger area, never committed; tier 1 the first 300 | the realistic photo rows (crops per photo, EXIF mix, camera encoders) | private only (section 6.8) |
| set A | 2,000 pinned COCO (`coco_bench_2000.json`) | 2,000 | small-image rows, CPU-route controls | public |
| COCO val2017 | 5,000 | 5,000 | AP and recall when boxes move | public |
| Open Images faces | `openimages_faces.pin.json` | the pin | SCRFD and ArcFace rows | public |
| LFW, CALFW, CPLFW | `lfw.pin.json`; the local copies of [labelled_benchmarks.md](../../research/labelled_benchmarks.md) section 3 | verification pairs | ArcFace parity after alignment changes (verification accuracy not lower) | public (local copies only; never redistributed) |
| CARS196 test, SOP test | [labelled_benchmarks.md](../../research/labelled_benchmarks.md) section 5 | test splits | task-level check for any change that moves PE vectors: Recall@1 not lower by more than 0.2 points | public |
| OCR set | the pinned text set of B0 and 5c | the pin | P-ocr | public |
| EXIF fixtures | tags 1-8, progressive, truncated, CMYK, 4:4:4/4:2:2/4:2:0, restart markers | about 40 files, generated by a script from public images | decoder coverage | public |
| public vector sets | not needed: these prototypes produce vectors; parity is measured on their own outputs (cosine, P-knn). The clustering plan owns vector-set work | - | - | - |

Copying set P (the owner runs it, or a lane with the owner's yes): a script in the prototype folder
takes `--src`, `--dst`, `--n 1000`, `--seed 20261104`, selects files by a seeded shuffle of the
sorted relative paths, copies with `shutil.copy2`, verifies sha256 after copy, and writes the
manifest to `--manifest-out` (a path the owner chooses outside the repository). The source is read
only and never modified or deleted; the scratch copy is not deleted without the owner's yes.

Sampling rules: the same files in the same order for every arm of one comparison; the input page
cache dropped before each repetition (`posix_fadvise(DONTNEED)`, as 07 M-4); the first 50 images
of each repetition are warm-up, excluded.

### 6.4 Metrics

Per row (one prototype configuration on one set):

| Metric | How |
|---|---|
| throughput | images/s over the counted images, median of the repetitions, with min and max |
| latency | per-image wall ms p50 and p99 (and per-stage for the pipeline prototypes) |
| GPU time per image, by stage | Nsight Systems CUDA trace on one tier-1 repetition: kernel ms per stage, copies, syncs, `cudaMalloc` count, idle gaps; `nsys stats` exported to CSV |
| GPU work, not just busy | DCGM or NVML sampling every 100 ms: `SM_ACTIVE`, `SM_OCCUPANCY`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, utilization (for comparison with older rows), SM clock, power |
| host CPU per image | process CPU-s from `/proc/<pid>/stat` deltas (all threads), and per-thread maximum share (the GIL question) |
| GPU memory | peak device MiB per process (NVML), CuPy pool high-water mark |
| parity, pixels | the tensor gate against the reference input (best shift (0,0), mean abs diff < 0.5 grey, max <= 8), or byte identity where the math is unchanged |
| parity, boxes | every matched box within **1 px at the original resolution**; P-box >= 0.98, P-item >= 0.99 |
| parity, embeddings | PE crop and frame cosine **median >= 0.999**, p01 >= 0.995; P-knn overlap@10 >= 0.95 |
| parity, faces and OCR | face boxes within 1 px, ArcFace cosine p01 >= 0.995, P-face 1.0; P-ocr >= 98 % exact, CER not worse |
| coverage | decode status per file (ok, declined with reason, failed), EXIF orientation equal on tags 1-8 |
| environment | commit, versions (driver, CUDA, TensorRT, every venv package from `pip freeze`), GPU UUID and clocks, power cap, host load at start and end, foreign GPU processes |

The references are the product's own outputs where a stack exists (`ref-b5a` after 5a), else a
prototype reference run of the shipped algorithm in the venv (PR-3 arm "today"), captured once per
set and reused.

### 6.5 Repetition and spread

- Tier 1 (screen): one repetition on the tier-1 slice; drops arms more than 10 % behind the best.
- Tier 2 (decide): 3 counted repetitions after one warm-up, candidate and control only, same
  session, alternating order (A B A B A B) to spread drift.
- Spread = (max - min) / median of a row. An arm wins only when its median gain exceeds 2x the
  larger spread of the two arms and every parity gate of its kind passes (07 M-5).
- A repetition is noisy when the 1-minute load average exceeds 16 at its start or a foreign process
  appears on the card; noisy repetitions are repeated once, then reported as noisy.
- One factor at a time; no full factorial; one confirmation run per adopted bundle.

### 6.6 GPU etiquette

- GPUs 0 and 2 only; GPU 1 never (other workloads own it). One GPU job at a time from this program;
  two-card prototypes (PR-7) are the only ones that use both cards, in one agreed slot.
- Before each session: ask the peer sessions and wait for the answer; `uptime` and `nvidia-smi`
  record; stop at once when a peer asks.
- Serialize with a shared lock file all jobs agree on (`flock <scratch>/locks/gpu<N>.lock <command>`,
  `docs/research/CLAUDE.md`); no `pgrep -f`, `pkill -f`, `sleep` or `until` loops in briefs or
  scripts; long runs in the background with a completion notification.
- A prototype never touches the default compose project, the live stack's containers or its
  models directory.
- Each session has a written GPU-hour cap (section 6.9); past the cap the session stops and reports.

### 6.7 Stop rules (per prototype)

1. **Not binding**: the stage it targets is under 10 % of the measured GPU-ms or host-ms per photo at
   the bulk point; stop and record.
2. **No gain**: tier-2 median gain under 2x the spread or under the prototype's own bar.
3. **Parity**: a gate fails, one documented fix is tried, it fails again.
4. **Two implementations forever**: adopting it needs a second preprocessing path that cannot be
   held to byte identity or the tensor gate against the first.
5. **Budget**: it needs more than twice its GPU hours or a second unplanned session.
6. **Superseded**: a cheaper line reaches the same ceiling (for example the second card).

A stopped prototype's code stays in the research folder with its result note (it is the evidence);
nothing from it enters `src/` or `models/`.

### 6.8 Recording results

- Raw files: `artifacts_local/research/gpu_pipeline/<prototype>/<run_id>/` (environment JSON,
  per-repetition JSON, nsys reports, DCGM samples); never bulk-deleted (the owner's research-data
  rule).
- **Public-dataset results** (sets A, B, COCO, Open Images, LFW, CARS196, SOP, the OCR set): a dated
  note in `docs/research/gpu_pipeline_prototypes.md` (Setup, Method, Results, Findings, Open issues,
  Reproduce, as `docs/research/CLAUDE.md` asks), a row in `docs/research/README.md`, and small files
  that back a published number under `docs/benchmarks/research_gpu/` with `MANIFEST.sha256`.
- **Set-P results stay private**: their numbers, ratios, file names and locations are written only
  to the private ledger and `artifacts_local/`; tracked text names set P only and, with the owner's
  yes (J3), whether a set-P bar was met. Nothing derived from set P goes into a commit message, an
  issue comment, a gist or `docs/`.
- Every decision taken from a prototype is written back into this file (section 4 verdict and
  section 6.9 outcome column) in the same docs commit as the note.

### 6.9 The prototype list

GPU hours are estimates for tier 1 + tier 2 + one confirmation. "Window" refers to section 8.

| ID | Plain name | Question | Arms | Sets | Pass bar (adopt) | Stop | GPU h | Old IDs |
|---|---|---|---|---|---|---|---:|---|
| PR-0 | Harness and the honest GPU number | does "94 % busy" hide idle SMs; what does each stage cost in GPU ms on the shipped algorithm | none (instrumentation): environment capture, DCGM/NVML sampler, nsys wrapper, parity adapters to `parity_gates.py`, the set-P copy script | 50 images of each set | the harness reproduces the shipped kernels' output byte for byte and the SP0 stage split within 2x spread | - | 1 | E-1 (part) |
| PR-1 | Decoder matrix | which decoder per size class and resource | D-a ctypes `GPU_HYBRID` lanes (import of `nvjpeg_lanes.py`); D-b nvImageCodec newest at batch 1, 8, 64 with `GPU_ONLY` and `HYBRID_CPU_GPU`, `:fancy_upsampling=1`; D-c DALI `mixed` (pipeline and dynamic mode), `jpeg_fancy_upsampling=True`, `hybrid_huffman_threshold` 0 and default; CPU libjpeg-turbo (simplejpeg) and Pillow `draft` (5c-K4 column) | set B, set P, set A, OI 1,024 px, EXIF fixtures | keep D-a unless an arm is >= 1.15x photos/s per GPU-ms **or** -30 % GPU-ms per photo at equal photos/s, with the tensor gate, EXIF 1-8 equal, the same declined-file set (no silent acceptance, D3) | stop rules 1-3 | 3 | E-2, H-3 |
| PR-2 | Resize and crop ops | can CV-CUDA replace the custom kernels | K-a CuPy kernels (import of `prep_kernels.py`); K-b CV-CUDA `pillowresize` / `resize` / `copymakeborder` / `laplacian` / `resize_crop_convert_reformat` on `ImageBatchVarShape`; DALI `fn.resize` with the three parity settings | set B, set P, set A | byte-identical **or** (tensor gate and P-box and P-pe against the venv reference) **and** GPU-ms per photo not higher; a library that passes replaces the custom code (5b-G2 rule) | 1-4 | 1.5 | 5b-G2 pre-work |
| PR-3 | Resident-frame pipeline, direct | what one process can do with no Triton: decode, letterbox, detector and PE through the TensorRT Python runtime, crops, blur, crop encode; per-image worker threads vs an asyncio loop with K CUDA streams; 1 vs 2 processes on one card | today's algorithm on the PR-1/PR-2 winners; streams 2, 4, 8; workers 8, 16, 32; pinned host buffers on/off | set B, set P | establishes the direct ceiling D (photos/s, SM active, GPU-ms per photo); explains the 1-instance cap (32 vs 48) by trace: if 1 process with 2+ streams reaches 2-instance speed, one instance can replace two (half the memory) | 1, 5 | 4 | E-3, E-7 (part), H-1, H-13 |
| PR-4 | Through-Triton cost | how much the Python backend, BLS and decoupled responses cost against PR-3 | the shipped `gpu_image_prep` on the stock pinned Triton; the same with the PR-1/PR-2 winners; 1 and 2 instances; record whether async BLS works in decoupled async `execute` on the pinned release | set B, set P | gap G = 1 - (PR-4 photos/s / PR-3 photos/s) at equal parity; **G >= 0.25 with SM active < 85 % opens X-c (C++ backend)**; otherwise X-c closes for this hardware | 1, 2 | 3 | E-3, E-6 (MPS check, conditional), H-1, H-6 |
| PR-5 | Faces on uint8 and the GPU | uint8 inputs with normalization in the graph for SCRFD and ArcFace; SCRFD anchor decode and NMS in the graph; batched CV-CUDA `warp_affine` alignment against `cv2.warpAffine` | ONNX Runtime CPU equality first (no GPU), then TensorRT in the venv | Open Images faces, set A persons, LFW pairs | outputs equal within FP32 rounding offline; face boxes within 1 px, ArcFace p01 >= 0.995, P-face 1.0, LFW verification not lower; host ms per face -30 % | 1-3 | 1.5 | E-4, E-9, H-7 |
| PR-6 | OCR recognition tail | argmax in the recognition graph; padding-aware CTC on bucketed widths | ORT then TensorRT in the venv; same crops as the 2026-11-02 bucket run | OCR set, 1,500 COCO val2017 text regions | P-ocr >= 98 % exact, CER not worse; recognition GPU-ms and output bytes per crop lower by > 2x spread | 1-3 | 1 | 5c-O2/O3 pre-work, H-10 |
| PR-7 | Two cards | one process per card NUMA-pinned (`numactl --cpunodebind --membind`) against one process driving both cards | PR-3 pipeline (direct), then PR-4 (through Triton) for the winner | set B 500 x 4 passes, set A 2,000, set P | >= 1.8x one card on set B and set A; pick by photos/s, then p99, then operability; below 1.6x name the binder | 1, 2, 5 | 2 (both cards, one agreed slot) | E-11, H-5 |
| PR-8 | Interactive under bulk | single-image p99 on the stock Triton under bulk image-prep load: `priority_levels`, a rate-limiter resource, a reserved instance | PR-4 setup | set B bulk, set A interactive | interactive p99 <= 2x idle with bulk >= 95 % of solo | 2 | 1.5 | E-12, H-11 |
| PR-9 | Opt-in INT8 PE, second round | SmoothQuant and per-layer `auto_quantize` with FP16 allowed | the #213 protocol unchanged (disjoint calibration set, same parity tool) | pinned 2,000 (12,790 crops + 2,000 frames), CARS196 and SOP Recall@1 | every #213 gate **and** >= 1.15x PE embeddings/s; Recall@1 within 0.2 points; else the INT8 line closes for good | 2, 3 | 3 | E-10, H-9 |

Total about 21.5 GPU hours. The earlier E-5 (CUDA graphs on small engines) stays in 5a-S5 and the
post-tag wave; E-8 (batched GPU decode for small images on stateless routes) is the 5b-W1 bake-off
fed by PR-1 and PR-4; E-13 (bundle confirmation) is the R-final row and the post-tag confirmation.
Mapping of the earlier H-n: H-1 PR-3/PR-4; H-2 5b-W1; H-3 PR-1; H-4 5b-C1; H-5 PR-7; H-6 PR-4
(conditional); H-7 PR-5; H-8 5a-S5; H-9 PR-9; H-10 PR-6 and 5c-O1; H-11 PR-8; H-12 closed (X-g);
H-13 PR-3; H-14 the image move (D-38), rows only.

## 7. Production design for the winners on the new foundation

What ships once the prototypes have decided; boxes that depend on a prototype say which, and keep
today's form if it fails.

```
API replicas (stateless, no GPU)              Celery `jobs` workers (no GPU; bulk_ingest_run, reprocess,
  interactive routes, priority 1                directory_ingest; gpu_affinity any_card; priority 2)
        \                                         /
         \   router (S-15): serving profile servers, fewest outstanding, model ready in serving_reports;
          \  one image's calls stay on one server
           v                                     v
+---------------- Triton server per card (S-16; one card today, two after PR-7) -----------------+
| image-prep executor (X-b; X-c only if PR-4 opens it): one decoupled model, per-image workers,   |
|   byte budget and in-flight bound from the serving profile (S-5), counted fallbacks (S-18/S-20)  |
|   decode once: decoder per PR-1 (D-a today), EXIF on the GPU, uint8 frame resident for the      |
|   request only                                                                                  |
|   views cut per consumer from the full frame, each from its engine's input contract (S-1):      |
|     detector letterbox 640 -> detector (end2end)                                                |
|     PE whole-frame square + per-detection squares 336 -> PE (FP16; INT8 opt-in row, PR-9)       |
|     faces branch: SCRFD 640 uint8 (in-graph decode + NMS, PR-5) -> aligned 112 (CV-CUDA,        |
|       PR-5) -> ArcFace                                                                          |
|     OCR branch: detector view -> line crops -> recognition (async calls 5c-O1, argmax PR-6)     |
|     work crops capped at crop_max_side -> JPEG (host threads, or nvImageCodec per 5b-C1)        |
|   branches run only when the resolved plan (S-9, 5d) asks for them                              |
| engines of the load list only (S-5), memory measured per engine (S-4)                           |
+-------------------------------------------------------------------------------------------------+
Small images: CPU decode in the caller (3 MP gate), the same geometry module (S-17), uint8 tensors.
Curation workers (VLM, segmenter, region stage): read capped work crops through the storage seam
(S-10); no frame crosses a process boundary.
```

| Concern | Rule | Grounding |
|---|---|---|
| Memory owner | Triton owns all GPU memory on its card; the API and the `jobs` workers own none | B4; one context per card (gpu_decode_plan 5.5) |
| Job semantics | progress, cancel, resume and fencing come from the `jobs` row (02b); the executor is stateless per request, so a cancelled job simply stops sending images | 02b sections 2-3 |
| Back-pressure | the executor's in-flight bound and byte budget answer `gpu_memory`/`over_budget`; the job counts them in `JobPerfSummary` and takes the CPU route for that image (a counted route, not a degraded mode: D-69 concerns dependencies, not routes) | SPF.1 |
| Fail closed | a server whose `serving_reports` lack the model, an engine whose contract does not match its config, or a stamp mismatch refuses that model at load (READY never reached), and the router does not send to it | #415, S-3, S-13 |
| Interactive vs bulk | Triton `priority_levels` on the engines and the executor; the serving profile holds the levels | PR-8, 5a-S6 |
| Two cards | one server per card, NUMA-pinned, identical load lists; the router balances; nothing crosses the socket | PR-7, M-a |
| Configuration | the serving profile (S-5) replaces the per-profile JSON and the installer's `config.pbtxt` edits; `openprocessor gpu plan` renders it from `gpu_devices` (S-6) and measured engine memory (S-4) | S-5, S-6 |
| Observability | stage counters per server, DCGM profiling fields, per-job summaries | S-18..S-23 |

## 8. Landing lanes

| Wave | Lane (plain name, branch) | Files | Red test (fails before) | Green |
|---|---|---|---|---|
| prototype window PW-0 (owner question J1; GPU 0, then GPU 2, during W2-W4 when the stack's GPU windows are idle) | prototype harness and prototypes, `research/gpu-pipeline-proto` | `scripts/research/gpu_pipeline/*`, `scripts/research/gpu_pipeline/requirements-gpu-research.txt`, `docs/research/gpu_pipeline_prototypes.md`, `docs/research/README.md` | the harness self-test: the venv reproduction of the shipped kernels is byte-identical to `prep_kernels.py` on 20 fixtures, and the parity adapter fails on a planted 2 px box shift | PR-0..PR-6 rows recorded; verdicts written back here |
| W2.1 | schema and database layer (lane 1A, `b2/db-core`) | `src/db/models/training.py` (S-1..S-4), `src/db/models/ops.py` (S-6, S-7), `src/db/models/config.py` (S-5 check, S-14 table), `src/schemas/models.py` (`EngineInputContract`, `EngineOutputContract`), the `0001_control_plane` revision | `tests/db/test_schema_matches_plan.py` asserts the new columns, the four-part unique key and the UUID key; a model row without `input_contract` is refused; a contract with an unknown `resize` value fails validation | same tests green |
| W2.1 | storage seam (lane 1E, `b2/storage-seam`) | `src/clients/storage/` key builder (S-10) | a capped and an uncapped crop of one box get different keys | green |
| W2.2 | config store (lane 2C, `b2/config-pg`) | `serving_profile` kind and its pydantic body; the default profile generated from today's `config_templates/profiles/*.json` in the 0.6.0 upgrade step | a `serving_profile` with a `project_id` is refused; the generated default equals today's values (one server, load lists, instance counts) | green |
| W2.2 | job runtime (lane 3A, `b2/job-runtime`) | `src/config/job_kinds/` (`gpu_affinity`), `JobPerfSummary` model | a kind without `gpu_affinity` fails the kind-table test; a perf summary with an unknown route name fails validation | green |
| W2.3 | export and registry (lanes 5B, 5D) | `export/` writers of S-1..S-4, S-13 | an export whose ONNX input dtype differs from the declared contract fails the job with `contract_mismatch` | green |
| W2.4 | scale-out compose and monitoring (lane 6B) and beat (6A) | `docker-compose.yml` Triton service shape (S-16), DCGM field list (S-22), `op.periodic.serving_inventory` (S-14) | the compose test finds a Triton service with a fixed `container_name`; the beat test finds no serving inventory task | green |
| W4 | worker knob and pools (lane 4B) | `src/clients/triton_client_manager.py` router (S-15), `op_triton_router_outstanding` | with two fake servers, 100 concurrent calls split within 10 %; a server whose report lacks the model gets no call; one image's follow-up calls stay on its server | green |
| W5a | baseline (`ref`, `sweeps`) | records only: DCGM fields and GPU UUIDs in M-3/M-7; nsys from the host | - | the 5a rows carry SM active and tensor active |
| W5b | decode once (`decode`, `prep`, `routes`, `codec`) | adopt PR-1/PR-2 winners in `models/gpu_image_prep/1/` behind the gate; S-17 geometry module; S-18..S-20 metrics; X-c plan file only if PR-4 opened it | the 5b tests as written in 07b, plus: the geometry copy in the model directory equals `src/domain/image_geometry.py` | 07b gates |
| W5c | crops, OCR | PR-6 outcome into 5c-O2/O3; S-10 used by `crop_sizing` | 07c tests | 07c gates |
| W5d | processing plans, faces on the GPU | PR-5 outcome into 5d-P5; S-9, S-21 | 07d skip proofs; a plan without `faces` makes no SCRFD call | 07d gates |
| post-tag research wave RW (milestone per OQ-1) | two cards (PR-7) into production; interactive priorities (PR-8); INT8 opt-in (PR-9); CUDA graphs on small engines (E-5); MPS only on PR-4 evidence (#403); the alternative-server head-to-head under its own rule (#231) | serving profile with two servers, compose second Triton service, router defaults | two-server router and compose tests; the opt-in INT8 engine is a separate `model_engines` row that no default profile selects | E-13 confirmation rows |

## 9. Risks and stop rules

| Risk | Mitigation |
|---|---|
| the seams add columns no code reads in 0.7.0 (dead schema) | every seam has a writer and a reader in 0.7.0 (S-1..S-4 written by export, read by the CPU route's contract test and `gpu plan`; S-5 read by the router with one server; S-6 written by workers, read by reservations); a seam without a 0.7.0 reader is dropped from this list before W2.1 |
| prototype numbers do not carry into the product (venv vs container differences) | same TensorRT and CuPy versions as the image; PR-4 measures the product path on the stock image; adoption gates run on the B5 stack |
| research outruns correctness work | PW-0 touches no product file; it runs only on idle GPU time with peer consent |
| a shared, noisy host | tier rules, noisy-repetition rule, DCGM clocks recorded |
| library defaults silently change pixels (DALI chroma, EXIF) | explicit settings, tensor gate first, EXIF fixtures |
| custom ctypes nvJPEG code ages with CUDA releases | PR-1 re-tests nvImageCodec at batch over 50 every image move; J8 |
| a native backend becomes a maintenance burden | X-c opens only on the PR-4 bar and with its own plan file; stop rule 4 |
| set-P data leaks into public text | section 6.8 rule; the naming scan; the copy manifest lives outside the repository |
| two-card runs take GPU 2 from a peer | PR-7 in one agreed slot; it can fall back to the 5a-S3 two-card record |
| INT8 drift into a default | separate engine row, opt-in only, never in an installer profile |

Programme-level stop rules: a prototype line ends by section 6.7; the whole research line on photos
on one card ends when PR-0 shows SM active above 85 % with PE at least 70 % of GPU ms per photo
(then only the second card and precision remain, both already planned).

## 10. Owner questions

| ID | Question | Recommendation |
|---|---|---|
| J1 | Run the venv prototypes PR-0..PR-6 in a prototype window during W2-W4 (GPU 0, GPU 2 when agreed; about 15 GPU hours), before the tag, instead of all research after the tag (MQ-9)? They touch no product file and decide 5b-5d's choices | yes; PR-7..PR-9 stay after the tag |
| J2 | Prototype code tracked in `scripts/research/gpu_pipeline/` with exact pins (not a gitignored folder) | yes (reproducibility rule) |
| J3 | Set P: a seeded 1,000-photo copy on local NVMe; may public text say only "the set-P bar was met / not met", with no numbers? | yes to the copy; yes to met/not met only |
| J4 | Put seams S-1..S-17 into the v0.7.0 schema and services now (first Alembic revision), even where the winner lands later | yes; each has a 0.7.0 reader (section 9) |
| J5 | If PR-1 shows CPU Huffman frees enough GPU time, allow a deployment to choose CPU or GPU Huffman by serving profile (never adaptive at run time) | yes, as a serving-profile field with one default chosen by measurement |
| J6 | A C++ Triton backend in the repo if PR-4 opens X-c (build toolchain in the Triton image, C++ review) | yes, only on the bar, with its own plan file |
| J7 | One Triton per card as the two-GPU default after PR-7 | decide after PR-7; the seams make either answer a configuration change |
| J8 | Keep the custom ctypes nvJPEG lanes until a library reaches GPU Huffman at real batch sizes with equal pixels | yes; re-test in PR-1 and at every Triton image move |
| J9 | DCGM profiling fields in the monitoring profile (the exporter may need extra capabilities) | yes for bench and monitoring stacks; off in the minimal install |
| J10 | INT8 second round (PR-9), 3 GPU hours, after the tag | yes, once; it closes the line for good if it fails |
| J11 | Milestone for the post-tag wave: reopen `v0.9.0` or a new one | a new milestone, for example `research-gpu-pipeline` |
| J12 | Hardware with a JPEG decoder (A100/H100 class) in scope? | no; plan for GA102; S-6's `hw_jpeg_decoder` keeps the door open without code |

## 11. Required changes to other plan files (for the integration agent; this lane edits none of them)

| File | Section | Change |
|---|---|---|
| [02a_postgres_schema.md](02a_postgres_schema.md) | 4 (config) | add `serving_profile` to the `config_documents.kind` CHECK; change the deployment-wide CHECK to `(kind IN ('vlm_endpoint','serving_profile')) = (project_id IS NULL)`; add table `serving_reports` (S-14) with retention 1 day like `config_runtime_reports` |
| 02a | 5 (jobs) | add table `gpu_devices` (S-6); change `gpu_reservations` primary key to `gpu_uuid` referencing `gpu_devices` (S-7) |
| 02a | 7 (models) | add to `model_engines`: `input_contract`, `output_contract`, `build_profile`, `build_settings`, `build_stamp`, `onnx_sha256`, `engine_sha256`, `device_mib_measured`, `max_batch`, `opt_batch`, `cuda_graphs`, `parity_ref`, `parity_report_key`, the `precision` CHECK; unique key `(triton_model_name, trt_version, gpu_arch, build_profile)` (S-1..S-4) |
| 02a | 9 (retention) | rows for `gpu_devices` and `serving_reports` |
| [02b_jobs_locking_events.md](02b_jobs_locking_events.md) | 2.1 | kind-table column `gpu_affinity` (`none`, `any_card`); `any_card` for `bulk_ingest_run`, `reprocess`, `directory_ingest` (S-11) |
| 02b | 2.2 / 3.1 | `jobs.params.plan`, `plan_sha256` (S-9); `JobPerfSummary` in `jobs.result` for the three kinds (S-8) |
| 02b | 7.3 | note that no GPU-prep queue exists in 0.7.0 and that device-scoped queues are derived from `gpu_devices` when declared (S-12) |
| [02c_migration_ops_tests.md](02c_migration_ops_tests.md) | 2 (0.6.0 upgrade) | generate the default `serving_profile` from the installed `config_templates/profiles/*.json` and `config.pbtxt` instance counts |
| 02c | 5 (storage seam) | crop key with the size variant (S-10) |
| [03a_stage1_foundation.md](03a_stage1_foundation.md) | lane 1A, 1E | the schema seams and their red tests (section 8 rows W2.1) |
| [03b_stage2_control_data.md](03b_stage2_control_data.md) | lane 2C | the `serving_profile` kind and its tests |
| [03c_stage3_job_runtime.md](03c_stage3_job_runtime.md) | lane 3A | `gpu_affinity`, `JobPerfSummary` |
| [03e_stage5_gpu_queues.md](03e_stage5_gpu_queues.md) | lanes 5B, 5D; GPU worker shape | export writes S-1..S-4 and fails on `contract_mismatch` (S-13); `make_app(device_uuid)` (S-12); reservations by UUID |
| [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md) | 6A, 6B | `op.periodic.serving_inventory` (S-14); Triton per-card service shape (S-16); DCGM profiling fields (S-22) |
| [05_b4_stateless_api.md](05_b4_stateless_api.md) | lane 4B | the router (S-15) and `op_triton_router_outstanding` (S-23) |
| [06_b5_overview.md](06_b5_overview.md) | 5 (M-1, M-3, M-7, M-9) | M-1: set P is allowed for prototypes and private gates under section 6.8 of this file; M-3/M-7: record DCGM `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, clocks, power and GPU UUIDs; section 7 library table: add the nvJPEG facts of section 3.2 (batched GPU Huffman over 50, scaled and batched ROI decode hardware-only) and DALI's `jpeg_fancy_upsampling` default |
| [06a_b5_baseline.md](06a_b5_baseline.md) | 5a-R0, 5a-S4 | read PR-0/PR-1 records first; 5a-S4's nvJPEG check at 20 MP is PR-1's set B row if PW-0 ran |
| [06b_b5_decode_once.md](06b_b5_decode_once.md) | 5b-P1 | Nsight Systems is removed from the Triton image: mount the host `nsys` read-only into the bench stack or use PR-3/PR-4 traces; add DCGM fields to the report |
| 07b | 5b-G1 | add S-17 (one geometry module copied into the model directory, with the identity test) |
| 07b | 5b-G2 | CV-CUDA evaluation is PR-2; the lane reads its record before writing kernels |
| 07b | 5b-G3 | set `jpeg_fancy_upsampling=True` and `adjust_orientation=True` explicitly; verify ragged JPEG batches on the pinned DALI backend (its README says homogeneous shapes only) |
| 07b | 5b-G4 | replace the rule with the PR-4 bar (gap >= 25 % against PR-3 at SM active < 85 %) |
| 07b | 5b-C1 | nvImageCodec `Encoder` refuses chroma-subsampled image objects: encode from the RGB crop array; record the encoder id used |
| 07b | metrics | rename `op_ingest_decode_total` and `op_reprocess_decode_total` to `op_decode_route_total{route, path}` in 5b-D1 (S-18), no alias |
| [06c_b5_crops_ocr_faces.md](06c_b5_crops_ocr_faces.md) | 5c-K4 | GPU-route reduced decode is impossible on the A6000 (nvJPEG scaled decode is hardware-backend only); the task is the CPU-route measurement |
| 07c | 5c-K1 | crop cache keys with the size variant (S-10) |
| 07c | 5c-O2, 5c-O3 | read PR-6's record first |
| [06d_b5_processing_plans.md](06d_b5_processing_plans.md) | 5d-P5 | CV-CUDA `warp_affine` evaluation is PR-5; uint8 SCRFD/ArcFace inputs and in-graph decode/NMS adopted only on PR-5's pass; `op_plan_branch_total` (S-21) |
| [00_master_plan.md](00_master_plan.md) | 5 (GPU windows), 10 | add the prototype window PW-0 (J1) as a side track on idle GPU time during W2-W4, never blocking a wave; the post-tag research wave with PR-7..PR-9 |
| [decision_register.md](reference/decision_register.md) | 6 | register J1-J12 as open decisions with this file's recommendations; MQ-9 amended if J1 is yes |
| [traceability.md](reference/traceability.md) | issue map | #40, #407, #403, #231, #268: point at PR-n of this file; earlier E-n/H-n retired into PR-n (section 6.9 mapping) |
| [standards_and_libraries.md](reference/standards_and_libraries.md) | library register | rows: nvJPEG through ctypes (custom, justified against nvImageCodec, re-tested per J8); nvImageCodec; DALI; CV-CUDA (0.19.0, licence read at adoption); DCGM exporter profiling fields; Nsight Systems on the host |
| [risks_cleanup_host.md](reference/risks_cleanup_host.md) | 1 (host and GPU rules) | PW-0 etiquette (section 6.6) and the set-P copy rule (section 6.3) |
| [01_package_pins.md](01_package_pins.md) | new subsection | the research venv's exact pins are owned by `scripts/research/gpu_pipeline/requirements-gpu-research.txt`, not the product manifests; `tensorrt-cu13` and `cupy-cuda13x` there move with the Triton image |
