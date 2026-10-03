# Triton pipeline optimization plan: crop at model size, decode once, stay on the GPU

Status: plan only (nothing implemented). Issue: #40. This is the single
consolidated plan; it replaces the earlier private design notes. A fresh agent
with no memory should be able to implement it from this file alone.

Paths are relative to the repo root. File:line references were re-verified
against main at `7d6123da` (after generic-detector W0-W3) and will drift;
re-find by symbol before editing. Env vars written as `OP_...` all exist in code
today. New settings are written in lowercase (`crop_max_side`) as proposals:
the env names are chosen at implementation time, and each wave that adds one
must add it to `env.template` and `tests/test_env_surface.py`. Routes are
written without a method token unless they exist (the docs checker validates
`METHOD /path` strings).

Sibling plans this one must compose with (read them, do not duplicate them):

- `docs/design/generic_detector_and_selective_embedding_plan.md` (#52): full
  80-class default detector, every detection stored, embedding policy
  `all | selected | lazy`. W0-W3 are on main (`embedding_state`,
  `seed_from_detector`); W4-W9 (policy store, ingest gating, embed-missing
  reprocess, lazy triggers) are being built by other agents.
- `docs/design/sam3_full_image_detection_plan.md` (#30): a full-image SAM 3
  runner whose hits become normal items.

## 1. Decisions already made

1. **Baseline first.** No optimization code merges before Wave 0 numbers exist
   on both fixed sets (section 5). Every later wave re-runs the same sets.
2. **Crops are cut once, from the original-resolution frame, at the size the
   consuming model needs** (the owner's key observation, section 2.1). Embeddings
   still come from the original pixels (never from the downscaled detector
   input); what changes is that the cut is resized immediately to the model's
   input size instead of being carried around at full resolution.
3. **Decode once.** One decode per image feeds the detector letterboxes, the
   crops and the blur metric. The whole-frame embedding keeps its reduced
   decode until an owner-approved embedding-space migration.
4. **GPU where it measures faster, CPU fallback forever.** The CPU path stays as
   a supported backend (CI, CPU-only installs, small GPUs). The GPU path is
   selected by a setting and defaults off until its gates pass.
5. **Few, large BLS calls.** Variable-count work (N crops, M text lines) lives
   in one GPU-resident Python BLS model; fixed chains use ensembles. Never one
   BLS call per crop.
6. **Parity by construction before parity by tolerance.** The first wave of
   each stage must be bit-identical to today's output where that is possible;
   anything that changes numerics is gated by the accuracy table in section 7
   and ships behind a flag.
7. **No data migration in this plan.** Nothing here changes a stored embedding
   space. The optional squash/full-resolution-whole-frame migration (Wave 10)
   is separate, evaluated, and needs an explicit owner yes.
8. **Public data for public numbers.** `docs/PERFORMANCE.md` carries COCO
   numbers only. Private-photo numbers stay in private notes; the repo records
   only pass/fail against gates.
9. **Small, independently shippable waves**, each red-first tested with a
   mutation check, behind a rollback switch.

## 2. Current state (re-verified against main)

### 2.1 Front and center: crops are cut and carried at full resolution

On a 20 MP photo the detector sees a 640 px letterbox, PE sees 336 px, the
region detector sees 640 px, SAM 3 sees 1008 px, and OCR detection sees at most
960 px on its long side. Yet every crop is cut from the full-resolution frame
and then processed, encoded, stored and shipped at that full size:

| # | Stage | Code | What happens at full crop size |
|---|---|---|---|
| 1 | Cut | `src/services/curation/ingest_index.py:88` (`crop_pil`), `:115` | `PIL.Image.crop` copies the full-resolution region per item |
| 2 | PE embed | `ingest_index.py:118-119`; `src/clients/pe_encoder.py:359` (`embed_crops`); `src/services/detection/pe_preprocess.py:34` (`resize_crop_rgb`) | `np.asarray` of the full crop (second copy), then `cv2.resize` from full size to 336 (INTER_LINEAR from, say, 2700x1900: an 8x downscale) |
| 3 | Blur | `ingest_index.py:156-158,182`; `src/services/detection/crop_quality.py:68-100` | RGB to BGR reversed-stride full-frame copy, CV_64F Laplacian on the full frame, then per crop |
| 4 | Crop cache | `ingest_index.py:185`; `src/services/curation/source_image_cache.py:130` (`write_crop_cache`), quality 90 at `:43` | **JPEG-encodes the full-size crop** and writes it to tmpfs for every stored item |
| 5 | Cache cap | `src/config/curation.py:154-158` (`crop_cache_dir`, 3 GiB), `ingest_index.py:224` | Full-size crops fill the cap fast; evicted crops fall to the disk path below |
| 6 | Cache miss | `src/services/curation/crop_bytes.py:55-97` (`crop_jpeg_from_disk`) | Re-decodes the entire source frame, crops, re-encodes q90 |
| 7 | Worker load | `scripts/curation/worker/state.py:345`, `scripts/curation/worker/runner.py:761-768` | Reads the full-size crop JPEG into memory once per task |
| 8 | Region detector | `src/services/detection/cascade_detect.py:486` (`detect`), `:776` (`_preprocess`) | PIL-decodes the full-size crop JPEG, letterboxes to 640, FP32 over gRPC |
| 9 | Segmenter | `scripts/curation/worker/client.py:372-389` (`segment_multi`), `runner.py:1192`; `docker/segmenter/main.py:203` | **Base64 of the full-size crop JPEG in a JSON body**; the service decodes it, then the model resizes to 1008 |
| 10 | VLM | `runner.py:1447-1453` (`CombinedCrop(jpeg_bytes=t.crop_jpeg)`), `src/services/labeling/vlm_labeler.py:1385-1418` (`draw_region_overlay`, `_b64_jpeg`) | Overlay re-decode/re-encode, then **base64 of the full-size JPEG** to the VLM, whose own processor downsizes it |
| 11 | OCR | `cascade_detect.py:1122` (`read_lines`), `:1263-1306` (`_preprocess`) | Sends `original_image` as **FP32 at full crop resolution** to the OCR BLS (about 10.6 MB for a 1100x800 crop, about 62 MB for 2700x1900) |
| 12 | Sub-crop re-pass | `scripts/curation/worker/cascade.py:181-196` (`_crop_region_jpeg`) | Decode the full-size crop again, crop, re-encode |

Contrast: the API-side VLM preview already shrinks to a 224 px long side
(`crop_bytes.py:115`, `load_vlm_item_jpeg` at `:128`; used at
`src/routers/curation/vlm.py:237`). The worker's real VLM and segmenter path
does not. That is the cheapest win in this plan.

#### Byte and time accounting, one 20 MP frame (5472 x 3648)

Worked crops (RGB uint8, `W x H x 3`):

| Item crop | Full-res RGB | PE input (336 FP32 / u8) | Region det. (640 FP32 / u8) | SAM 3 (1008 u8) | OCR `original_image` FP32 |
|---|---:|---:|---:|---:|---:|
| medium, 1100 x 800 | 2.6 MB | 1.35 MB / 0.34 MB | 4.9 MB / 1.2 MB | 3.0 MB | 10.6 MB |
| large, 2700 x 1900 | 15.4 MB | 1.35 MB / 0.34 MB | 4.9 MB / 1.2 MB | 3.0 MB | 61.6 MB |

With the all-80-class default (about 7 items per image on COCO; more on busy
photos) these multiply by the item count, so a frame with 7 large crops moves on
the order of 100 MB of crop pixels through full-size copies before any model
runs. Today, the model consumes 0.3 to 3 MB of it.

Measured stage times (24 MP frames, 3 items, CPU contended; ratios matter more
than absolutes), from the private 2026-09-24 measurement that this plan is
based on:

| Stage | ms |
|---|---:|
| PIL full-res decode + EXIF | 309-336 |
| letterbox 640 + 1280 (PIL resize from full-res) | 391 |
| primary infer / secondary infer (when configured) | 62 / 387 |
| crop + PE preprocess (3 crops) | 65 |
| PE crop infer / whole-frame infer | 34 / 37 |
| whole-frame re-decode (cv2 1/8) + preprocess | 77 |
| full-frame blur copy + CV_64F Laplacian | 875 |
| per-crop Laplacian (3 crops) | 201 |
| crop-cache JPEG encode q90 (3 crops) | 85 |
| worker: crop JPEG re-decode + letterbox 640 | 228 per crop |
| worker: OCR preprocess, FP32 full crop | 448 |
| worker: OCR BLS on a whole-item crop (2616x2344 to 4963x3531) | 841-1779 |

About 2.0 s of CPU per frame goes to pre- and post-processing for roughly
150 ms of GPU compute. Most of that CPU is full-resolution pixels no model
needs. End-to-end, on public COCO (small images) a 2,000-image run ingested at
13.4 images/s, while a private archive of 12-20 MP photos ran at about
2.5 images/s on the same stack: the cost scales with pixels, not with detections.

Derived (to be measured in Wave 0, not yet verified): crop-cache pressure. At
7 items per image and a few hundred kilobytes per full-size crop JPEG, a 3 GiB cache
holds on the order of a thousand images of crops. A larger ingest evicts
crops the worker has not reached yet, and each eviction triggers the full
re-decode in row 6.

**Why COCO alone cannot show this.** COCO images are about 640 x 480, so crops
are already near model size and the full-size penalty barely exists. The
headline waves (Wave 3 onward) therefore need a high-resolution set for their
gates (section 5.1 and open question Q1).

### 2.2 Other waste, current as of main

| ID | Waste | Evidence |
|---|---|---|
| W-a | Decode runs on the event loop, serially, in the batch prefill | `src/services/curation/ingest_batch.py:57-96`; ingest runs concurrently only up to `OP_MAX_INGEST_CONCURRENCY` (`src/services/curation/ingest.py:111-112`) |
| W-b | Full-res PIL decode, then PIL BILINEAR resize from full-res for each letterbox | `src/services/curation/ingest.py:134-141` (`_decode_image`); `src/services/detection/geometry.py:113-146` (`letterbox_to_square`) |
| W-c | FP32 NCHW detector inputs (4x the bytes of uint8) | `src/services/curation/ingest_detect.py:145` (`run_primary_batch`), `:320-380` (secondary) |
| W-d | When a secondary raw-head detector is configured: raw head plus feature map come back (about 39 MB per image measured) and NMS/RoI pooling run on the client | `ingest_detect.py:340-380`; optional, off in the generic stack |
| W-e | Whole-frame embedding is a second decode (cv2 1/8) and a separate PE call per image | `ingest_index.py:132-141`; `pe_preprocess.py:60-97`; `pe_encoder.py:396-428` |
| W-f | One PE crop call per image (no cross-image batching) | `ingest_index.py:119` |
| W-g | OCR recognition is one batch-1 call per text line inside the BLS; vocabulary-wide output copied to host | `models/ocr_pipeline/` (single instance); `models/paddleocr_rec_trt/config.pbtxt:38,45-46` |
| W-h | Queue delays of 15-25 ms on models that already receive client batches | `models/yolov11_small_trt_end2end/config.pbtxt:39-44`, `models/pe_image_encoder/config.pbtxt:37-38`, `models/paddleocr_rec_trt/config.pbtxt:45-46`, `models/mobileclip2_s2_image_encoder/config.pbtxt:29-30` |
| W-i | `--log-verbose=1` in the shipped compose file; no CUDA pool sizing flag | `docker-compose.yml:59` |

Already done (do not redo): the OCR worker payload channel order is BGR
(`cascade_detect.py:1263-1285`, the old RGB/BGR defect is fixed);
`embedding_state` and `seed_from_detector` exist
(`src/services/curation/embedding_state.py`, `src/routers/curation/class_seed.py`).
Not on main yet: the embedding policy store and ingest gating (#52 W4-W9). Until
it lands, `ingest_index.py:115-127` embeds every item and marks `failed` when the
encoder raises.

Measured Triton behavior to design around (2026-09-24, private measurement,
numbers only):

- PE is the GPU ceiling: about 170 img/s on one 48 GB card at batch 32 (a frame
  with k crops costs k+1 PE slots).
- DALI/nvJPEG on these Ampere GA102 cards has no hardware JPEG engine (hybrid
  Huffman on CPU): about 40-50 ms of CPU per 20-26 MP image versus 309-336 ms
  for PIL, 70-111 images/s per GPU depending on DALI threads.
- BLS GPU inputs cost one device-to-device copy plus an IPC round trip each;
  the default Triton CUDA memory pool (64 MB) is smaller than one decoded
  20 MP frame (60 MB) and falls back silently to pinned host memory or CUDA IPC.
- OCR recognition batch-8 in one call took 91.5 ms versus 175 ms as 8 sequential
  batch-1 calls (each waiting out the queue delay).
- An earlier independent benchmark harness measured DALI decode instances
  1/2/4/8 as 103/176/240/169 img/s at 32 clients (4 best, 9.3 GB GPU peak at 4).

## 3. How this composes with the other plans

| Concern | Rule |
|---|---|
| All-80-class default, about 7 crops per image | Crop cost scales with item count, so cut cost must be per-crop minimal. Wave 1 and Wave 3 pay off roughly 5x more under the generic default than under a narrow vehicle filter. |
| Embedding policy `all \| selected \| lazy` (#52) | The PE crop is cut and resized **only for items that will be embedded**. Items stored without a vector (`not_selected`, `lazy`) still get the small work crop for the cache, but no PE tensor. When the embed step (embed-missing reprocess, lazy trigger) runs later, it cuts from the source with the same shared crop function, so an item embedded later gets the same vector as one embedded at ingest. The cut function must take the item list after policy gating; do not embed-then-discard. |
| The VLM, review queue and clustering operate on the embedded working set (#52 P3) | VLM payloads use the small work crop (Wave 3); the work-crop size cap is the second cost knob next to the embedding policy. |
| SAM 3 full-image runner (#30) | Its input is a whole frame, not a crop. Whole-frame payloads use a configurable `image_max_side` (its plan default 1024), cut from the already-decoded array rather than re-decoding the file. Its hits become items through the same item writer, so they flow through the same crop/embed path and policy. Waves 1-3 here change only the crop and decode helpers it reuses, never its contract. |
| Region profiles and the segmenter crop path | Wave 3 changes the crop payload only (smaller JPEG); the contract (`crop_jpeg_b64`, normalized boxes) is unchanged because boxes are normalized to the crop frame. |
| Batch endpoint `/segment/batch` | Not used by the worker today; batching worker segmenter calls is out of scope here (see Wave 8). |

## 4. Target architecture

```
client --JPEG bytes (+ flags)--> API: hash + dedup (bytes only)
  CPU backend (always available):
    decode once (cv2, EXIF) -> uint8 frame array
      -> letterboxes for detectors (640 from the 1280 intermediate)
      -> per item: ROI view of the frame -> resize ONCE to each needed size
           PE 336 tensor (only if the item is to be embedded)
           work crop (max side = crop_max_side) -> JPEG -> crop cache
      -> blur on one gray plane
  GPU backend (selected by setting; Triton ingest_pipeline BLS):
    DALI nvJPEG decode (mixed) -> full_rgb uint8 [H,W,3] resident on GPU
      -> detector letterbox tensors (uint8) -> detector via BLS
      -> small heads to host (boxes/scores), NMS + IoU on CPU
      -> per-item ROI slice -> float only for the ROI -> interpolate to 336/640
         written into one preallocated [N,3,S,S] batch per consumer
      -> ONE PE call for whole frame + N crops; ONE region-detector call
      -> work crop encoded small (<= crop_max_side) for the cache
    returns kilobytes: boxes, vectors, strings, small crop JPEGs
  Worker / VLM / segmenter consume the small work crop. Text reading cuts a
  small region crop from the source at native resolution (it is small by
  nature) and never uploads a full item crop as FP32.
```

Design rules:

- **Crop sizes per consumer** (the smallest crop that preserves the
  information the model can use):

  | Consumer | Size | Source of the crop |
  |---|---|---|
  | PE crop embedding | 336 (resize shorter edge + center crop, as today) | ROI of the full-res frame |
  | Region detector | 640 letterbox | work crop (max side >= 640) |
  | Segmenter | up to 1008 on the model side | work crop |
  | VLM (worker) | `vlm_crop_max_side`, default tied to the API preview's 224 only after the agreement gate; see Q3 | work crop |
  | OCR detection | long side 640, range 320-960 (`cascade_detect.py:962-964`) | work crop |
  | OCR recognition lines (height 48) and region text | native source pixels | region sub-crop from the source |

- **Work crop** = the ROI resized so its long side is at most `crop_max_side`
  (proposed, default 1024 once gated; 0 = today's behavior, full size). It is
  the single artifact written to the crop cache and read by every worker stage,
  so cache hit and cache miss (`crop_jpeg_from_disk`) must produce the same
  shape: both go through one function.
- **Resident crops across models**: in the GPU backend the resized crops stay on
  the GPU between the PE call and the region-detector call (one BLS request,
  one ROI source tensor). The DALI hand-off and each BLS input incur one D2D
  copy, so the design batches per consumer and never calls per crop.
- **Static ensembles** for the stateless single-model routes
  (`JPEG -> DALI preprocess -> model`), no Python in the loop.
- **Crop kernels** (GPU): axis-aligned resize via torch (`F.interpolate`,
  antialias per reference) on a uint8 ROI slice converted to float for the ROI
  only. Do not use `torchvision.ops.roi_align` on the whole frame: it needs a
  float full frame (240 MB for 20 MP) and is not a PIL/cv2-equivalent resampler.
  Faces (affine) and text lines (perspective) are separate families and are out
  of scope until a face/OCR wave is approved (section 10, Q6).
- **Triton tuning**: short queue delays (1-2 ms) on models fed by client- or
  BLS-batched callers; preferred batch sizes matched to the measured knee; a
  second PE instance only if the baseline shows at least 10% throughput;
  `model_warmup`; `--log-verbose=0`; an explicit `--cuda-memory-pool-byte-size`
  once any GPU tensor crosses the Python-stub boundary; pinned pool shrunk once
  inputs are uint8/JPEG.
- **GPU constraints, stated generically.** A large (about 48 GB) card hosts the
  full model set plus several concurrent frames (about 140 MB steady, about
  280 MB peak per in-flight 20 MP frame with the uint8-plus-ROI design). A small
  (about 12 GB) card holds the CNN detectors but not the full set plus DALI plus
  BLS stubs; there use the CPU backend or the trimmed set. TensorRT plans are
  tied to the TensorRT version and GPU architecture (SM86 plans are portable
  across Ampere cards, but workspace sizing differs); never assume a plan built
  for one card is tuned for another.

## 5. Wave 0: baseline first (no behavior change)

### 5.1 The two fixed sets

| Set | Content | Reproducibility |
|---|---|---|
| A. Public COCO 4,000 | 4,000 images drawn with a fixed seed from COCO val2017 (5,000 images), license recorded per image | Committed manifest `scripts/datasets/manifests/coco_bench_4000.json` (image ids, sha256, bytes, width, height, license, seed). Built with `scripts/datasets/fetch_coco_subset.py` (`--val-only`, `--manifest`); it needs a uniform-random, all-class mode because its default selection is balanced over 10 classes, so add an option rather than a second script. |
| B. Private 4,000 | 4,000 photos of about 12-20 MP drawn with a fixed seed from a private archive | Manifest (relative path, sha256, w, h, bytes, EXIF orientation) kept in gitignored `artifacts_local/bench/`, never committed. Built by `scripts/bench/bench_sets.py build-private --root <dir> --n 4000 --seed 42 --min-mp 12`. The images are read in place, read-only. |

Plus small guard lists derived from B (200-image parity subset; 20 images with
EXIF orientation other than 1, synthesized from copies in `artifacts_local/`
if the archive has too few) and from A (200-image parity subset).

Set A shows the public, reproducible effect (many small images, many items).
Set B shows the crop-size effect (section 2.1). A public high-resolution set
for the Wave 3 gate is open question Q1.

### 5.2 What to measure (per stage, per set)

Stages: decode, detect (letterbox plus infer), crop (cut plus resize), JPEG
encode, host-to-device (derive from Triton `compute_input`), resize, embed,
region/segmenter, VLM, cluster, OpenSearch write. For each: wall ms and CPU ms
(`time.thread_time`), bytes moved (client counters on every Triton request,
crop JPEG bytes, VLM and segmenter payload bytes), GPU utilization (mean) and
peak memory (pynvml at 10 Hz), storage per image (store bytes, crop-cache bytes,
vectors), items per image, and the crop-cache hit/miss counts
(`src/services/curation/crop_bytes.py` `cache_stats`).

### 5.3 Scripts (existing and proposed)

| Script | Status | Purpose |
|---|---|---|
| `scripts/bench/ingest_cost_probe.py` | exists (pure arithmetic, tested by `tests/test_ingest_cost_probe.py`) | Storage and call-count table. Extend: accept measured per-image crop bytes, work-crop bytes and VLM payload bytes so the table shows the crop-size effect |
| `scripts/bench/bench_sets.py` | new | `build-coco`, `build-private`, `verify <manifest>` (re-hash, fail on drift) |
| `src/utils/stage_timing.py` | new | Tiny opt-in recorder (a context manager keyed by stage name, a no-op unless enabled) used by the real code at the stage boundaries; unit-tested for the disabled no-op and for nesting |
| `scripts/bench/ingest_stage_timer.py` | new | In-process: runs the real ingest functions on N images (default 50) and writes the per-stage table as JSON and markdown. Uses the fake sinks in `tests/integration/ingest_fakes.py` for OpenSearch |
| `scripts/bench/run_ingest_benchmark.py` | new | End to end through the batch ingest route with a fresh project per run (project slugs are retired after delete, so use a new slug each time): images/s, p50/p99 per batch, API and Triton process CPU-seconds per image, GPU sampler, store and cache growth |
| `scripts/bench/triton_stats_delta.py` | new | Snapshot `/v2/models/stats` before and after a run; per-model deltas for queue, input, infer, output, average batch |
| `scripts/bench/worker_stage_probe.py` | new | Scrape the worker's Prometheus metrics (segmenter, VLM, region stage durations; `OP_SEGMENTER_REQUEST_*` histograms exist) before and after the worker drains; record crops/s |
| `scripts/bench/parity_capture.py`, `parity_compare.py` | new | Capture current outputs for the parity subsets (item docs without ids/timestamps, boxes, vectors keyed by `(sha256, box index)`, VLM labels, segmenter candidates), then compare against the gates of section 7 and exit non-zero on failure. `parity_compare` is the one piece that must be right: unit-test it on synthetic boxes, vectors and strings |
| `tests/test_bench_*.py` | new | Tests for the above pure logic |

Perf_analyzer points (from the Triton SDK image, `--network host`, gRPC,
`--measurement-interval 5000 --stability-percentage 10`) for each TensorRT model
at batch 1/8/16/32 by concurrency 1:16:*2 are recorded alongside, as a
micro-benchmark that tells you which model is the ceiling.

### 5.4 How to run (rules)

- Benchmark on a dedicated GPU that no other workload is using; record
  `nvidia-smi` and host load average before each run and discard runs with load
  above 8 (rerun on a quiet host).
- Host networking for the bench containers (port mapping proxies every byte).
- Use a locally-built image tag for test images (a watchtower-style updater can
  replace public `:latest` tags over local builds).
- Warm up: discard the first 100 images of each run (graph load, caches).
- **3 interleaved rounds** (A, B, A, B, A, B) per comparison; report the median
  and the range, never a single run.
- Run both sets: A for the public numbers, B for the crop-size numbers. Run the
  ingest at `OP_MAX_INGEST_CONCURRENCY` of 16, 32, 64 and keep the best.
- All raw JSON goes to `artifacts_local/bench/<wave>/<set>/<arm>/<round>.json`
  (gitignored). Never pipe benchmark output into the repo.

### 5.5 Recording before/after

- `docs/PERFORMANCE.md` gets a "Pipeline optimization" section with a table per
  wave: stage, before, after, ratio, **COCO set only**, plus the commit and the
  stack description (GPU model, versions). The existing "Ingest cost per image"
  section stays; extend it with the measured crop-byte columns.
- Private-set numbers are recorded in the private archive's notes. The PR body
  and `docs/PERFORMANCE.md` state only "private-set gate: pass/fail" and the
  ratio, never an image name, path or count that identifies the archive.
- Record the Triton, TensorRT, DALI, torch and driver versions with every
  table; engines are tied to them.

### 5.6 Verification checks to answer in Wave 0 (write results into section 11)

- C1: does a Triton ensemble run a step whose outputs are all unrequested?
- C2: does `async def execute` with `async_exec` fan-out overlap BLS calls in a
  non-decoupled Python model on the shipped Triton version?
- C3: BLS per-call overhead for GPU inputs of 1, 10 and 60 MB, with the CUDA
  pool at 64 MB versus 1 GiB.
- C4: do any `pinned system memory` or `CUDA IPC` fallback lines appear in the
  Triton log under load?
- C5: dynamic batcher behavior under bursts (first request's queue time with a
  short delay and a large preferred size).
- C6: does a DALI `mixed` output handed to a BLS create a second full-frame copy?
- C7: what image size does the deployed VLM's processor actually resize to
  (read its preprocessor config), and what is the segmenter's real input size?
  This sets the defaults in section 4.

Wave 0 is done when the scripts and manifests are merged, the baseline tables for
sets A and B exist, C1-C7 are answered, and the A-set baseline is in
`docs/PERFORMANCE.md`.

## 6. Waves

Every wave: feature branch off main, conventional commits, red-first tests (watch
each new test fail for the stated reason), a mutation check (break the rule,
watch the test fail, restore), `.venv/bin/pre-commit run --all-files`, the
whole `tests/` including `tests/test_naming_leaks.py` and `tests/test_docs_vs_code.py`,
the section 7 gates, a before/after run of both sets, merge with `--no-ff`.
Each wave names its rollback switch. Order: 0, then 1 to 4 in any order after 0
(they are independent), then 5, 6, 7, 8, 9, and 10 only on explicit approval.

### Wave 1: cut once, no copies (CPU, bit-identical)

Goal: remove the redundant full-size copies without changing any output.

- Files: `src/services/curation/ingest_index.py` (`crop_pil`, `index_items`),
  `src/services/detection/pe_preprocess.py`, `src/clients/pe_encoder.py`
  (`embed_crops`), `src/services/detection/crop_quality.py`,
  `src/services/curation/ingest_batch.py`, `src/services/curation/ingest.py`.
- Changes:
  1. Convert the decoded frame to a uint8 array once; per item take a **view**
     slice (same rounding and clamping as `crop_pil`) and call
     `resize_crop_rgb` on the view. Same operation on the same pixels, so
     embeddings are bit-identical; the PIL crop copy and `np.asarray` copy go.
  2. Blur: compute one gray plane (`cv2.cvtColor(rgb, COLOR_RGB2GRAY)`, no
     reversed-stride copy), per-crop variance on slices. Keep `CV_64F` so the
     stored `blur_lap_ratio` is unchanged unless the gate allows float32 (then
     `CV_32F` within the P-blur tolerance).
  3. Whole-frame PE from bytes already in memory in batch ingest: reduced decode
     once per image in the prefill, in a thread pool (cv2 releases the GIL), and
     all whole-frame vectors of a batch in one PE call. The reduced decode is
     kept because it defines the stored whole-frame space.
  4. Move the CPU pixel work off the event loop (`asyncio.to_thread`/the
     existing executor) in `_prefill_detections` and `index_items`; decode a
     batch concurrently.
  5. Skip the second per-image duplicate query when the batch msearch already
     resolved the hash (`ingest_batch.py:150` versus the per-image lookup in
     `ingest.py:340-400`).
- Tests (red-first): crop view equals `crop_pil` for edge cases (negative,
  out of bounds, one-pixel boxes); embeddings bit-identical to the old path on
  fixtures; blur ratios equal within the stated tolerance; a slow fake decode
  does not delay a concurrent health request; the batch resolves duplicates
  without a second lookup. Mutation: off-by-one in the rounding, drop the
  thread offload.
- Gates: P-box, P-pe bit-identical, P-pew bit-identical, P-blur, P-item.
- Acceptance: ingest CPU per 20 MP image at least 40% lower than baseline on set
  B; set A not slower; p99 per batch not worse by more than 10%.
- Rollback: revert (no data change).

### Wave 2: faster decode and letterboxes (CPU, parity-gated)

- Files: `src/services/curation/ingest.py` (`_decode_image`),
  `src/services/detection/geometry.py` (`letterbox_to_square`),
  `src/services/curation/ingest_detect.py`.
- Changes: `cv2.imdecode(..., IMREAD_COLOR)` (applies EXIF) then one RGB
  conversion; an ndarray path in `letterbox_to_square` using `cv2.resize`;
  build the 1280 letterbox from the frame and the 640 from the 1280
  intermediate. PIL BILINEAR downscales with antialiasing and cv2 INTER_LINEAR
  does not, so try `INTER_AREA` and pick by the box gate. A setting
  (proposed `ingest_decode`, `pil | cv2`, default `pil` until gated) selects
  the path.
- Tests: EXIF orientation 6 fixture rotates identically; ndarray letterbox
  within the tensor tolerance of the PIL path; a live-marked detector parity
  test. Mutation: skip the EXIF transpose.
- Gates: tensor-level, P-box, P-item on both sets and the EXIF list.
- Acceptance: decode plus letterbox CPU at least 50% lower; detector gates
  green; flip the default to `cv2` only after the gate.
- Rollback: set `ingest_decode` back to `pil`.

### Wave 3: crops at model size (the headline wave)

- Files: new `src/services/detection/crop_sizing.py` (the one function that
  turns a ROI into a work crop and into model tensors), `src/services/curation/ingest_index.py`,
  `src/services/curation/source_image_cache.py`, `src/services/curation/crop_bytes.py`
  (`crop_jpeg_from_disk`, `load_item_crop_jpeg`, `load_vlm_item_jpeg`),
  `scripts/curation/worker/client.py`, `scripts/curation/worker/cascade.py`,
  `src/services/labeling/vlm_labeler.py`, `src/services/detection/cascade_detect.py`
  (`_preprocess` for the OCR payload), `env.template`, `tests/test_env_surface.py`.
- Changes:
  1. `crop_sizing` returns the work crop: the ROI resized (cv2 `INTER_AREA`,
     antialiased) so its long side is at most `crop_max_side`, never upscaled.
     JPEG encode from the small array. `write_crop_cache` writes it. The disk
     fallback `crop_jpeg_from_disk` calls the same function so hit and miss are
     identical in size and bytes layout.
  2. Worker consumers read the small crop: the region detector, segmenter,
     VLM combined path and OCR detection all inherit it with no signature change
     (they take JPEG bytes). Add a VLM-specific cap (`vlm_crop_max_side`) applied
     where the payload is built, default from Q3.
  3. OCR upload: send `original_image` as uint8 HWC capped at 2x the detection
     size (about 1920 long side), and add a new uint8 input to the OCR BLS next to
     the FP32 one (keep the old input one release; the model branches on which
     input is present). Text reading of a located region cuts from the source at
     native resolution (the region box is small).
  4. The policy interaction: items not selected for embedding still get the work
     crop (cheap); the PE tensor is built only for items to be embedded.
  5. Settings are off (0 = today's behavior) until the agreement gate passes, then
     the default becomes 1024.
- Tests (red-first): the work crop long side is at most the cap; never
  upscaled; aspect preserved; cache hit and miss produce the same dimensions;
  payload bytes to the segmenter and VLM are lower for a large synthetic crop
  (byte-count assertion); with cap 0 output equals today's bytes (golden);
  region and OCR normalized boxes map back to source coordinates identically
  (`crop_norm_to_source_norm` round trip). Mutation: ignore the cap, upscale
  small crops, drop the aspect ratio.
- Gates: VLM label agreement, segmenter hit/miss and IoU, region-detector
  agreement, OCR exact match (section 7). **These need the high-resolution set
  (Q1)**; on COCO the cap never triggers, so the public gate is the unit tests
  plus byte accounting.
- Acceptance (set B): bytes to the segmenter and VLM per crop at least 5x lower
  for crops larger than the cap; crop-cache bytes per image at least 5x lower;
  cache hit rate not lower under the same ingest; worker crops/s not lower;
  all section 7 gates green.
- Rollback: set the caps to 0.

### Wave 4: Triton config and server flags (independent, no model rebuild)

- Files: `models/*/config.pbtxt`, `docker-compose.yml`, `docs/PERFORMANCE.md`.
- Changes: `max_queue_delay_microseconds` to 1000-2000 on the primary detector,
  PE, MobileCLIP and OCR recognition (callers already batch); `model_warmup` at
  batch 1 and the preferred maximum; `--log-verbose=0`; optionally a model load
  thread count; an explicit `--cuda-memory-pool-byte-size` and a smaller
  pinned pool once Wave 6 inputs are uint8 (do not change the pool before that).
  Verify burst behavior with C5 before shortening delays on models fed by
  independent single-image callers.
- Tests: compose contract test updated (`tests/test_compose_contract.py`);
  a config lint test that no model's delay exceeds a cap without a comment.
- Acceptance: on Triton stats the average queue time of the PE and detector
  models falls below 3 ms at the target concurrency; throughput not lower; p99
  not worse by more than 10%.
- Rollback: revert the config commit; reload the model.

### Wave 5: uint8 inputs and smaller outputs (TensorRT rebuilds)

- Files: `export/export_models.py` and the secondary/OCR exporters, new model
  version directories (`2/`), `src/services/curation/ingest_detect.py`,
  `src/services/detection/cascade_detect.py`, the OCR BLS model.
- Changes: uint8 HWC detector and PE inputs (normalize and transpose in the
  graph; 4x fewer bytes on the wire and host-to-device); recognition argmax in
  the graph (about 9,000x smaller output) with a smaller engine profile; when a
  secondary raw-head detector is configured, in-graph NMS and RoI pooling (output
  about 100x smaller); OCR BLS batches recognition by width bucket (one call per
  bucket) and runs two instances. Each export is a new version directory
  selected by `version_policy`.
- Tests: ONNX-level parity of preprocessing folded into the graph (graph output
  versus the numpy preprocessing on fixtures); CTC decode on indices equals CTC
  on the softmax argmax; width bucketing. Mutation: wrong normalization constant.
- Gates: P-box, P-pe, P-ocr, P-bb (backbone vectors, if applicable).
- Acceptance: wire bytes per image at least 60% lower; secondary per-image output
  time (when configured) under 2 ms; OCR BLS time per request at least 3x lower;
  Triton GPU memory not higher.
- Rollback: set `version_policy` back to version 1 and reload.

### Wave 6: GPU decode and static ensembles

- Files: new `export/build_dali_preprocess.py`, `models/image_preprocess_dali/`,
  `models/detect_jpeg/`, `models/embed_image_jpeg/`, `src/clients/triton_client.py`,
  the detect and embed routers, `env.template`.
- Changes: a DALI pipeline (nvJPEG mixed decode, EXIF on) emitting the detector
  letterbox tensors; two ensembles for the stateless routes; settings
  (proposed `triton_jpeg_ensembles`, default off). Letterbox arithmetic
  (pad split, rounding) must match the CPU path exactly; compute the box
  un-mapping on the client from the JPEG header (size plus EXIF orientation, no
  decode) with the same arithmetic and test equality.
- Bake-off before adopting (same model, set B, concurrency 8/32/128, 3 rounds):
  A = CPU decode (Wave 2), B = DALI ensemble with 1/2/4 instances, C = many-process
  CPU decode. Choose by images/s per GB of GPU memory and p99.
- Gates: tensor-level diff (best shift (0,0), mean under 0.5 grey level), P-box,
  P-clip for embeddings.
- Acceptance: stateless detect route at 32 clients at least 1.5x the CPU arm with
  p99 not worse; otherwise ship with the setting off.
- Rollback: setting off.

### Wave 7: GPU-resident `ingest_pipeline` (decode once, crops resident)

- Files: new `models/ingest_pipeline/` (Python BLS), `models/image_decode_dali/`,
  new client `src/services/curation/ingest_gpu.py`, `env.template`, a Triton
  image pin for torch if needed.
- Changes: a Python BLS that takes JPEG bytes plus flags and returns
  measurements only (boxes, scores, vectors, small work-crop JPEGs, blur); the
  client keeps every decision about identity, dedup, class mapping, policy and
  writes. Crops per Wave 3 sizes, cut on the GPU from a resident uint8 frame,
  one PE call for the whole frame plus N crops (chunked at max batch), one
  region-detector call, work crop encoded small. Honors the embedding policy:
  the crop-embed flag lists only the items to embed. Selected by a backend
  setting (proposed `ingest_backend`, `cpu | gpu`, default `cpu`).
- Server flags: size the CUDA pool from C3/C4 (about (BLS instances + DALI max
  batch) x 64 MB plus crop batches); set the torch allocator to expandable
  segments; release the frame reference at the end of each request so the pool
  buffer is reused.
- Tests: pure helpers factored out of the model file (flag decoding, fixed
  output schema with empty tensors for disabled flags, shared crop-box rounding
  function used by both backends); a live-marked CPU-versus-GPU parity test.
- Gates: P-box, P-item, P-pe (cosine versus the CPU backend), P-pew, P-blur.
- Acceptance: API CPU per image under 60 ms; wire bytes about the JPEG size;
  no pool-fallback log lines; frames/s bounded by PE, not decode. **Stop and
  escalate if the BLS arm loses by more than 15% to the CPU-batched client arm
  at the target concurrency.**
- Rollback: set the backend to `cpu`; the models stay loaded unused.

### Wave 8: segmenter and region stage on resident crops (optional)

- Depends on Wave 7 and on `docs/design/sam3_full_image_detection_plan.md`.
  Two independent options, each measured before adoption: (a) run the region
  detector inline at ingest on resident crops and seed the pending status with
  the hit (the worker's pending-verification path already accepts a
  source-frame region box; append to the provenance chain, never overwrite);
  (b) serve the segmenter from Triton as TensorRT (a separate measured feasibility
  doc exists privately: 205 to 86 ms per crop and about 6 GB to about 1.35 GB on a
  12 GB card; the FP16 attention trap needs BF16 attention and the text-only
  geometry path needs a static fast path). Option (b) needs its own plan file
  before work starts. Batching the worker's segmenter calls is a client change
  inside this wave.
- Gates: region hit/miss agreement at least 99% against the worker's own
  detections, box IoU at least 0.9; segmenter top-1 IoU mean at least 0.97 (p05
  at least 0.90), hit/miss agreement at least 97%.
- Rollback: setting off.

### Wave 9: tuning pass and GPU layout

- One knob at a time on the benchmark GPU: instances, max batch and preferred,
  queue delay, DALI instances and threads, BLS instances, CUDA pool, CUDA graphs
  for the small fixed-shape models. Keep a change only if the end-to-end images/s
  improves at least 3% with p99 not worse by more than 10%.
- Decide the production GPU layout by measurement with the VLM under
  representative load. Promote defaults (`ingest_backend=gpu`, `ingest_decode=cv2`,
  work-crop caps) only after the full gates pass on both sets.
- Acceptance (one large card, 3 to 7 items per image): the target is set from the
  Wave 0 baseline; the historical private estimate was at least 35 frames/s
  PE-bound, API CPU at most 60 ms per frame, Triton total GPU memory at most
  12 GB for the CNN set (plus DALI and BLS stubs as budgeted from C3).

### Wave 10: embedding-space migration (only on explicit owner approval)

Optional and last. PE squash (a direct 336 x 336 resize, upstream's default,
instead of shorter-edge plus center crop), whole-frame embedding from the full
decode instead of the reduced decode, and faces from the original resolution
are all space changes. They must ride **one** re-embed, with a version field on
the image and item docs, version-filtered kNN/near-duplicate/cluster queries, a
resumable backfill, and a re-derived near-duplicate cutoff (today whole-frame
near-duplicates use cosine at least 0.98). Adopt only if kNN precision@10 and
residual cluster purity regress by no more than 1 point. Until then the served
space stays center-crop and the reduced decode.

## 7. Accuracy-regression gates

Validation ladder (stop at the first failure): (1) tensor level, (2) model
output, (3) task level (only for deliberate changes). Compare against the
Wave 0 captures of the same parity subsets.

| ID | Output | Rule | Pass |
|---|---|---|---|
| T | any new preprocessing tensor vs its reference | best +-1 px shift search | shift (0,0), mean abs diff under 0.5 grey level, max at most 8 |
| P-box | detector boxes | greedy match at IoU at least 0.5 within class | at least 98% of images identical (IoU at least 0.95, confidence delta at most 0.02); zero unmatched boxes with confidence at least 0.5; class ids identical |
| P-item | items per image after registry mapping | count and class | identical on at least 99% of images |
| P-pe | PE crop embeddings vs reference path | cosine | bit-identical in Wave 1; Waves 3 and 7: p01 at least 0.995, median at least 0.999 |
| P-pew | PE whole frame | cosine | bit-identical until Wave 10; GPU decode: at least 0.99 else it rides Wave 10 |
| P-bb | backbone RoI vectors (if a secondary is configured) | cosine | median at least 0.99, p05 at least 0.97, else keep client pooling |
| P-clip | MobileCLIP image | cosine | p01 at least 0.995 |
| P-ocr | OCR lines | exact string match after matching boxes at IoU at least 0.5 | at least 98% exact; character error rate not worse |
| P-region | region hit and box | agreement; IoU | hit/miss at least 99%; IoU at least 0.9 |
| P-sam | segmenter | top-1 hit/miss; box IoU | hit/miss at least 97%; IoU mean at least 0.97, p05 at least 0.90 |
| P-vlm | VLM label vs the label on the full-size crop | class label and region-visibility agreement | at least 97% agreement on the same crops; any disagreement list reviewed by a human |
| P-blur | `blur_lap_ratio` | abs diff | at most 0.01 (Wave 1), at most 0.02 (Wave 7) |
| P-json | route JSON for waves that change no numerics | deep equality minus timing and ids | 100% |

Rules: gate on matched boxes and verdict counts, not raw confidence (GPU IDCT
differences move borderline confidences by 0.1 to 0.15). Parity references record
**today's** behavior; fixing a defect is a deliberate "changed" result with its
own evaluation, not parity. Benchmark gate for every wave: at the target
concurrency, throughput at least +5% versus the previous shipped state (Wave 9
knobs +3%), p99 not worse by more than 10%, no pool-fallback log lines, peak GPU
memory within the budget.

## 8. Risks

1. **BLS throughput ceiling.** Non-overlapping `execute`, a device-to-device copy
   per GPU input, public reports of BLS well behind ensembles. Mitigation: C2/C3
   in Wave 0, few large calls, 2-6 instances, a side-by-side arm against the
   CPU-batched client with a stop-and-escalate rule at 15%.
2. **Numerics drift.** PIL antialiased resize versus cv2 versus GPU interpolate
   versus nvJPEG IDCT can silently move boxes near thresholds and embeddings across
   the stored space. Mitigation: bit-identical first (Wave 1), a per-model
   reference choice (gate against today's outputs), the tensor-level rung before
   the model-level rung, and one bundled migration (Wave 10).
3. **DALI memory.** DALI instances inflate memory and it is not released (set
   `release_after_unload`); a DALI to BLS hand-off may add a second frame copy
   (C6). Mitigation: size instances by images/s per GB; cap in-flight frames
   structurally (BLS instances plus DALI max batch).
4. **CUDA pool fallback.** The 64 MB default is smaller than one decoded frame;
   exhaustion falls back silently and a public issue reports empty outputs.
   Mitigation: explicit pool size, a log-grep gate in every benchmark.
5. **TensorRT engine rebuilds.** Plans are tied to the TensorRT version and the
   GPU architecture; uint8 and argmax re-exports are new version directories with
   rollback by `version_policy`; every TensorRT bump means rebuild plus gates.
   Strongly typed networks (TensorRT 11) bake FP16 into the ONNX.
6. **Small versus large GPU.** The full model set plus DALI plus BLS stubs does not
   fit a 12 GB card; that card keeps the CPU backend or a trimmed model set.
   Measure on the target card; do not extrapolate.
7. **Accuracy loss from smaller crops.** A VLM or segmenter that benefits from
   detail may regress when the work crop is capped. Mitigation: cap defaults come
   from C7 and the P-vlm/P-sam gates; text reading uses native-resolution region
   crops, never the capped crop.
8. **Host contention** distorts CPU-arm numbers. Mitigation: interleaved rounds
   and the load-average guard.
9. **Scope.** Tempting adjacent defects (generic-route decode counts, faces at the
   original resolution, a no-op batch OCR in generic ingest) are separate `fix`
   commits with their own tests, not part of a wave.
10. **No data migration** in Waves 0-9: a rollback is a setting flip or a revert.
    Do not let a wave write a new stored field.

## 9. Rejected and why

| Idea | Why not |
|---|---|
| `torchvision.ops.roi_align` on the full frame | Needs a float full frame (240 MB per 20 MP frame) and is not a PIL/cv2-equivalent resampler; use per-ROI uint8 slice then float on the ROI only |
| DALI doing the per-crop work | One sample in gives one sample out; N crops need N copies of the input |
| TensorRT in-engine cropping of the full frame | A 20 MP dynamic shape makes the profile unwieldy; RoiAlign fits only the fixed small feature map |
| Client-side nvJPEG plus CUDA shared memory | Forces a GPU into the API container and pins API and Triton to one device; revisit only if DALI loses by more than 15% |
| DALI CPU decode | Measured slowest (28-54 images/s) |
| Decode at reduced DCT size for the crops | Saves only the IDCT (Huffman still scans the whole file) and breaks parity; fallback only for memory-bound cards |
| Computing blur at half resolution | Changes the stored `blur_lap_ratio` semantics; a GPU float32 full-resolution per ROI keeps them |
| Changing PE to squash now | A new embedding space; belongs to Wave 10 |
| A per-crop BLS call | Each GPU input costs a device-to-device copy plus IPC; batch per consumer |
| INT8 PE | Accuracy risk; consider only if PE-bound after every other lever |

## 10. Open questions (with recommendations)

| ID | Question | Recommendation |
|---|---|---|
| Q1 | What public high-resolution set backs the Wave 3 gates, since COCO is about 640 x 480? | Build a small (about 500) license-checked set of 12+ MP photos from a public archive with per-image license in a committed manifest and a fetch script; until it exists, gate Wave 3 on the private set (pass/fail only in public docs) plus synthetic large-image unit tests |
| Q2 | Default `crop_max_side`? | 1024 (keeps SAM 3's 1008 input close to native); decide after C7 and the agreement gate; ship 0 first |
| Q3 | VLM crop cap: tie to the 224 px API preview or higher? | Start at the VLM processor's real input size from C7 (do not guess); lower to 224 only if P-vlm at least 97% holds |
| Q4 | Compute the embedding crop at ingest for items outside the working set? | No: follow the embedding policy; build the PE tensor only for items to be embedded. Embed-later runs the same shared crop function |
| Q5 | Which GPU layout in production, and does the benchmark GPU stay dedicated? | Decide in Wave 9 from measured numbers with the VLM under load; keep a dedicated benchmark GPU until then |
| Q6 | Include faces at the original resolution and OCR line crops on the GPU? | Out of this plan; each needs its own plan file and a migration decision (Wave 10) |
| Q7 | Gate default flips on the private set only? | Yes for behavior, with a public pass/fail line and the COCO ratio recorded in `docs/PERFORMANCE.md` |
| Q8 | Run Wave 8(b) (segmenter on Triton) here or as its own plan? | Own plan file, started after Wave 7 shows the segmenter leg is a large share of worker time |

## 11. Verification log (Wave 0 fills this in)

- [ ] C1 ensemble skips unrequested steps:
- [ ] C2 non-decoupled `async def execute` overlap:
- [ ] C3 BLS per-call overhead, pool 64 MB vs 1 GiB:
- [ ] C4 pool-fallback log lines under load:
- [ ] C5 batcher delay under bursts:
- [ ] C6 DALI to BLS second frame copy:
- [ ] C7 VLM processor input size, segmenter input size:
- [ ] Versions recorded: Triton, TensorRT, DALI, torch, driver
- [ ] Baseline tables: set A, set B

## 12. References

- `docs/design/generic_detector_and_selective_embedding_plan.md` (#52) and
  `docs/design/sam3_full_image_detection_plan.md` (#30).
- `docs/PERFORMANCE.md` ("Ingest cost per image"), `scripts/bench/ingest_cost_probe.py`.
- Triton Python backend (BLS, DLPack, `FORCE_CPU_ONLY_INPUT_TENSORS`, one stub
  process per instance); Triton ensemble models; the DALI Triton backend
  (limitations, `release_after_unload`); DALI `fn.decoders.image`
  (`hybrid_huffman_threshold`, `adjust_orientation`); nvJPEG; TensorRT
  EfficientNMS and ROIAlign plugins; perf_analyzer; Model Analyzer.
- PE-Core `get_image_transform(center_crop=False)` (squash is upstream's default).
