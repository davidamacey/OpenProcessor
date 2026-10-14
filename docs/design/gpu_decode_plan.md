# GPU decode once and a GPU-resident ingest GPU decode (#217, WP-2.4)

Status: built and on by default above 3 MP (section 13 records the build and the A/B; the settings are in `src/config/curation.py`). The text before section 13 is the plan as written, when nothing in `src/` had changed. Issue #217 (WP-2.4 of
[v060_wave_plan.md](v060_wave_plan.md)), parent design Waves 6 and 7 of
[triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md), related #67
(request-driven plans), #219 (work crops, WP-4.1), #220 (faces and OCR, WP-2.5). Written
2026-10-12 against `origin/main` `54fcfc89`; measured with the PE plan of `ref-u8fp16`, before the
rotary-layout engine of #227 (17-25 % faster PE), which lowers the PE share of section 7.5 and so
raises the GPU ceiling; the decode and GPU decode costs do not depend on it.

A fresh implementation agent executes this file alone: section 9 lists the work items in order
with files, interfaces, tests and gates; section 10 the verification commands. File and line
references were checked against `54fcfc89` and drift; re-find by symbol.

## 0. Summary

- **Photos are capped by CPU JPEG decode.** Ingest decodes every image twice on the CPU (Pillow in
  full, OpenCV at 1/8 for the whole-frame vector): 0.50 CPU-s per 18 MP photo. 46 processes doing
  nothing but that reach 26.3 photos/s, which is the measured photo ingest rate (24-28). COCO frames
  cost 7.4 ms and are bound by the PE engine, not the host.
- **GPU decode cuts the decode's host cost 3.7x, not to zero.** The A6000 has no hardware JPEG engine
  and nvImageCodec's GPU-only backend decodes nothing here, so the Huffman stage stays on the CPU:
  134 ms per photo, 2.8 ms per COCO frame. One decoder instance saturates at about 35 photos/s; scale
  by instances.
- **Through the stack**, a GPU decode (decode, letterbox, whole frame, blur on the GPU) cut API CPU
  per photo by 65 % (1.10 to 0.36-0.41 CPU-s) and host busy from 80 % to 30 %, but images/s stayed at
  23-27 because the float32 GPU kernels took about 15 ms of GPU per photo and the GPU became the limit.
  Cheaper kernels (half-precision resize input, convolution blur: 8 ms instead of 11 ms of GPU) gave
  25.5 against 26.8 photos/s on the CPU path: the GPU stays the limit until the 3 ms budget is met.
  With kernels inside a 3 ms budget the estimate is about 50 photos/s on one A6000 (1.9x)
  and about 80 on two (3x, host-bound). COCO frames lose 14 % with the GPU decode: they stay on
  the CPU path.
- **Numerics.** The GPU resizes pass the tensor gate (letterbox and crop squares within 1-4 grey
  levels, no shift) and the blur sums are bit-identical for the same pixels; the decoder alone moves
  the detector: P-box 0.87-0.93 against today's references (noise floor 0.99), so the `precision`
  parity profile fails (P-box, P-item, P-pe, P-knn, label vote). Accuracy does not change: COCO AP
  0.4376 against 0.4377-0.4379, kNN precision against ground truth unchanged, cluster purity within
  noise. Adopting GPU decode is a one-time numerics change that needs an owner decision (D1).
- **Resize chain (R1).** Every model input is resized from the full-resolution decoded frame;
  cascading from the detector letterbox wrecks the crops on photos (P-pe median 0.87) and moves the
  whole-frame vector away from today's space. The whole frame keeps today's space by emulating the 1/8
  DCT decode (cosine median 0.998-0.999).
- **Correctness hazards measured:** default chroma upsampling of every GPU decoder differs from
  libjpeg (fix: nvImageCodec `fancy_upsampling=1`, then 0.01 grey levels); torchvision's GPU decode and
  DALI's default ignore EXIF orientation; all GPU decoders return an image for truncated or corrupt
  files that Pillow rejects (route by an end-of-image check; the CPU verdict stands).
- **Where:** a Triton Python-backend GPU decode (`gpu_image_prep`, nvImageCodec plus custom kernels,
  sequence batching to keep the frame between the detector and the crops), not 32 API workers each
  holding a CUDA context (that ran the 48 GB card out of memory). **Size gate:** GPU from 3 MP up
  (measured cut-offs 2.8-23 MP depending on cards and detections per image). **Batching:** keep the
  PE and detector settings (measured flat); the GPU decode sends a request's whole-frame and crop
  squares as one PE tensor in chunks of 32.

## 1. Question, rules, inputs

The owner's direction (2026-10-12): GPU decode is the next large speed-up now that the CPU work is
done (uint8 inputs, event-loop offload, 2-GPU PE at 1.91x, the rotary-layout PE engine). He wants
(a) decode on the GPU, (b) each model's input resized on the GPU to its size and handed over in the
order the models run, (c) automatic batching and scaling. Added requirements:

| ID | Requirement | Where answered |
|---|---|---|
| R1 | Zero-copy resize chain: decode once at full resolution, keep the pixels on the GPU, derive each model input in model order (detector, PE whole frame, PE crops from the full-resolution frame using the detector boxes, then blur, crop JPEGs, other consumers); state which tensor every resize reads, its interpolation, the parity effect of resizing from an already-resized tensor (cascading), and count every remaining copy | sections 2, 3, 6.5 |
| R2 | Embedding batching: the PE crops of an image (about 6.4) go to Triton as one well-sized request; Triton settings justified by measurement; say what the GPU-resident path changes | section 8 |
| R3 | A CPU/GPU size gate with a measured cutoff | section 7.3 |
| R4 | Combine experiments into one session instead of repeated long runs | section 4.1 |

Decision rule (from the work package): recommend GPU decode only if end-to-end images/s on the
photo set and the total CPU improve materially and the parity and AP gates hold (or the visible
differences are quantified and small); say plainly what COCO-size frames gain; state the
end-to-end ceiling from the measured stage costs; a "large photos only" or negative answer is
acceptable with the routing threshold measured.

Data: public only. Set A is the pinned COCO set (`scripts/datasets/manifests/coco_bench_2000.json`,
640 x 480 median, 0.27 MP). Set B is the pinned public photo set
(`scripts/datasets/manifests/commons_hires_500.json`, 12-24 MP, median 17.7 MP, 10.4 MB per
file). Two Open Images samples fill the sizes between: **OI** = 60 images of the Open Images
`oi_plates_300.json` pin as served by the dataset bucket (1,024 px long side, 0.5-1.0 MP, median
0.78 MP) and **OIO** = 60 Flickr originals of the same pin (CC BY 2.0, 0.26-17.9 MP, median
5.2 MP). The decoder-coverage set (section 6) is generated from one set B photo by
`gpu_decode_bench.py make-formats`.

## 2. Dataflow today: which stage needs which pixels, at what size, in what order

Per image on `POST /ingest/batch` and `/ingest/upload` (the batch path decodes all images of a
request first, then runs the detector over the whole request, then indexes image by image):

| # | Stage (order) | Code | Reads | Produces | Size | CPU ms per image, frame / photo (WP-1.2 table) |
|---|---|---|---|---|---|---|
| 1 | Hash | `ingest._imohash_bytes` | JPEG bytes | imohash | n/a | under 0.5 / 28 |
| 2 | Decode 1 | `ingest._decode_image` | JPEG bytes | Pillow RGB frame, EXIF applied (`ImageOps.exif_transpose`, which also copies the frame) | full | 4.9 / 304 |
| 3 | Detector letterbox | `geometry.letterbox_to_square` via `ingest_detect.run_primary_batch` | frame (2) | uint8 `[1,3,640,640]`, Pillow `BILINEAR` (antialiased on downscale), fill 114, paste at `int(pad)` | 640 | 5.1 / 93 |
| 3b | Secondary letterbox (only with a secondary profile) | same, `run_secondary_raw_batch` | frame (2) | `input_size` letterbox | profile | off in the generic stack |
| 4 | Detector | Triton `yolov11_small_trt_end2end` | letterbox (3) | boxes in network pixels, un-mapped with `undo_letterbox` | | GPU 1-5 ms |
| 5 | Item crops | `ingest_index.crop_pil` | frame (2) + boxes (4) | Pillow crop per item, `round()` rectangle, zeros outside the frame | full-resolution crop | 0.5 / 8 |
| 6 | PE crop squares (items the embedding policy selects) | `pe_encoder._preprocess_crops` -> `resize_crop_rgb` | crops (5) | uint8 336 x 336: shorter edge to 336 with OpenCV `INTER_LINEAR` (no antialias), centre crop | 336 | 9.8 / 6.8 |
| 7 | PE crop embeddings | Triton `pe_image_encoder` | squares (6) | 1,024-d vectors | | GPU 5.5 ms each |
| 8 | Decode 2 (whole frame) | `pe_preprocess.whole_frame_rgb[_from_bytes]` | JPEG bytes again (file or upload bytes) | OpenCV `IMREAD_REDUCED_COLOR_8`: a 1/8-scale DCT decode (80 x 60 for a COCO frame), then `INTER_LINEAR` to shorter edge 336 (an **upscale** on frames), centre crop | 336 | 8.6 / 188 |
| 9 | PE whole-frame embedding | Triton `pe_image_encoder` | square (8) | 1,024-d vector | | GPU 5.5 ms |
| 10 | Full-frame blur | `ingest_index._full_frame_blur` -> `crop_quality.image_lap_var` | frame (2) | grey plane (OpenCV fixed point), 3 x 3 Laplacian, exact integer variance | full | 4.6 / 252 (with 11) |
| 11 | Box blur | `_box_blurs` -> `crop_lap_var` | frame (2) + boxes | same per clamped box rectangle | full-resolution box | in 10 |
| 12 | Crop cache JPEG | `source_image_cache.write_crop_cache` | crops (5) | Pillow JPEG q90 of the full-size crop to tmpfs | full-resolution crop | 3.4 / 23 |
| 13 | Documents and bulk write | `item_doc`, `_bulk_index` | vectors, boxes, blur | OpenSearch | | 3 / 10 |

Consumers outside ingest that need pixels later: the region stage and VLM (worker) read the crop
JPEGs of row 12 (WP-4.1 shrinks them), a cache miss re-decodes the source (`crop_bytes.crop_jpeg_from_disk`),
the embed pass (`reprocess_embed`) re-decodes and re-cuts crops from stored boxes, and the core
routes (`/detect`, `/embed`, `/faces`, `/ocr`, `/analyze`) decode with OpenCV or Pillow in
`src/utils/image_processing.py` and `src/clients/triton_client.py`. Faces and OCR are not called
by ingest (WP-2.5 owns them).

So a photo is entropy-decoded twice today (rows 2 and 8), and three full-resolution passes run on
the CPU after the decode (rows 3, 10, 11). On 12-24 MP photos rows 2, 3, 8 and 10 are 837 ms of the
1,090 ms of API CPU per photo that the stage timers attribute (WP-1.2). On a COCO frame the same rows
are 23 ms of about 60.

Copies per photo today (18 MP, 54 MB RGB): decode buffer, the `exif_transpose` copy, the
`np.asarray` for blur, the grey plane and the 16-bit Laplacian, the crops, plus a second
decoder's buffer; host to device only the uint8 model inputs (1.2 MB letterbox, 0.34 MB per PE square).

## 3. Target dataflow: decode once, the resize chain, every copy counted (R1)

Order of the chain, each step with the tensor it reads. "Original" is the full-resolution decoded
frame `F` (HWC uint8, on the GPU).

| Step | Tensor | Read from | Operation | Interpolation | Parity reference | Measured effect (sections 6.3, 6.5) |
|---|---|---|---|---|---|---|
| 0 | `F` | JPEG bytes (host) | nvImageCodec decode, EXIF orientation applied, libjpeg-style (fancy) chroma upsampling | n/a | Pillow decode | mean 0.03 grey levels, max 3 on COCO (3 % of pixels differ) |
| 1 | `L` detector letterbox `[3,640,640]` | `F` | resize longer side to 640, pad 114 at `int(pad)` | antialiased bilinear (Pillow's `BILINEAR` on downscale) | `letterbox_to_square` | tensor gate T passes on 100 % of images (mean 0.02-0.05 grey levels, max 4) |
| 2 | detector | `L` | TensorRT, same engine | | | P-box 0.87 against the CPU path (noise 0.994); almost all of it from step 0 (section 6.3) |
| 3 | `W` PE whole-frame square `[336,336,3]` | `F` | 8 x 8 block mean (the 1/8 DCT decode the product uses), then shorter edge to 336, centre crop | box filter, then bilinear without antialias (OpenCV `INTER_LINEAR`) | `whole_frame_rgb` | P-pew median 0.998, p01 0.991 on set A (gate for GPU decode 0.99); p01 0.984 on photos (section 6.5) |
| 3' | rejected: `W` from `L` (cascade) | `L` inner region | shorter edge to 336 | antialiased bilinear | | P-pew median 0.82 (same as from `F`: cascade and original agree, median 1.000, p01 0.970), so it is not today's space |
| 3'' | rejected: `W` from `F` directly | `F` | shorter edge to 336 | antialiased bilinear or area | | P-pew median 0.82 on set A: a new embedding space (the product upsamples an 80 x 60 thumbnail) |
| 4 | `C_i` PE crop squares `[N,336,336,3]` | `F` cut by the detector boxes (row 5 rectangles) | shorter edge to 336, centre crop | bilinear without antialias (`INTER_LINEAR`) | `resize_crop_rgb` | same boxes: P-pe p01 0.996, median 0.9994 on set A |
| 4' | rejected: `C_i` from `L` (cascade) | `L` cut by the boxes scaled into the letterbox | same | same | | P-pe p01 0.929 against 0.996 from `F` on the same boxes: small objects lose their detail in the 640 letterbox |
| 5 | blur sums | `F` (grey plane once) and the clamped box rectangles of `F` | OpenCV `RGB2GRAY` fixed point (weights 9798/19235/3735, 15 bits), 3 x 3 Laplacian, integer sums | exact | `image_lap_var`, `crop_lap_var` | bit-identical for the same pixels (unit tests); P-blur moves only through the decoder |
| 6 | crop cache JPEGs | `F` cut by the boxes (full size today; work-crop size after WP-4.1) | JPEG encode | | Pillow q90 | not measured (cache bytes only; the worker re-reads them) |

Rule that follows from the cascade measurements: every model input is resized from `F`, never
from another model's input. The detector letterbox is the largest model input but it is padded and
already downsampled; the whole frame and the crops are read from `F` (the whole frame through the
8 x 8 block mean that reproduces today's thumbnail). The only shared intermediate worth keeping is
the grey plane for the blur sums.

Copies in the target, per image (photo, 18 MP): host to device the JPEG bytes once (10.4 MB
against 54 MB of RGB), one decoded frame on the GPU, the grey plane (18 MB uint8) for the blur
sums, the letterbox and squares written straight into preallocated batch tensors; device to host
only boxes, vectors (4 KB each), blur sums (3 integers per box) and the crop JPEGs. Where the front
end runs outside Triton (section 7.1, stage 1) the letterbox (1.2 MB) and squares (0.34 MB each)
cross back to the host for the gRPC request: two small copies per tensor that a Triton-resident
GPU decode removes.

## 4. Measurements: method

| Item | Value |
|---|---|
| GPU | RTX A6000, GPU 2 (300 W cap, SM86, no hardware JPEG engine), PCIe 3.0 x16. GPU 0 (other A6000) and GPU 1 (RTX 3080 Ti) were not used: other projects hold them |
| Host | 2 x Xeon E5-2680 v3, 48 logical cores, shared with other projects (load average 9 to 39 at the end of the runs, the CPU pool probes push it themselves) |
| Images | `davidamacey/openprocessor:0.5.0` (API: Pillow 12.3.0 on libjpeg-turbo 3.1.4, OpenCV 4.10.0 on libjpeg-turbo 3.0.3, torch 2.14.1+cu130, torchvision 0.29.1) and `davidamacey/openprocessor-triton:0.5.0` (Triton 2.70, DALI 2.1.1, nvImageCodec 0.7.0.49, nvJPEG 13.2, torch 2.12.1, torchvision 0.27.1, CuPy 13.6). No package was installed into either image; the parity run mounted the Triton image's own nvImageCodec build into the API container |
| Engines | the `ref-u8fp16` plans (detector uint8, PE uint8 FP16, PE plan sha256 `a4feb7df...`) |
| Isolation | own containers and compose project `gpudec`, loopback ports 4910-4937; `nvidia-smi pmon` on GPU 2 for the whole session: only this work's processes ran on GPU 2 during every timed run (another project's `trtexec` started at 09:24:14, 3 s after the timing session released the lock); timed commands under `flock /tmp/ortloom-bench.lock` |
| Combined sessions (R4) | one timing session (CPU decoders, GPU decoders, decoder threads, contention: 36 minutes, one lock hold), one stack session (baseline and spike sweeps back to back), one untimed parity session (all variants in one capture per set) |
| Tools | `scripts/bench/gpu_decode_bench.py` (`cpu`, `gpu`, `pe-load`, `formats`, `make-formats`), `scripts/bench/gpu_decode_parity.py` (`capture`, `compare`), `scripts/bench/detector_quality.py` (AP), `scripts/bench/parity.py` (stack gates), `scripts/bench/ingest_sweep_suite.py` (stack sweeps) |

Raw JSON stays in the scratch directory of the run (section 12); the tables here are the record.

## 5. Measurements: decode and resize cost

### 5.1 CPU decoders (single thread, ms per image, median; API image)

| Class (median MP) | Pillow + EXIF (product decode 1) | Pillow DCT 1/8 | OpenCV full | OpenCV 1/8 (product decode 2) | product pair | product pair, 46 processes: images/s |
|---|---:|---:|---:|---:|---:|---:|
| A, COCO (0.27) | 5.3 | 2.2 | 3.7 | 2.1 | 7.4 | 979 |
| OI (0.78) | 10.1 | 3.4 | 5.8 | 2.9 | 13.0 | 851 |
| OIO (5.2) | 68.1 | 15.5 | 37.1 | 15.6 | 83.7 | 124 |
| B, photos (17.7) | 369 | 130 | 283 | 133 | 502 | **26.3** |

- On photos the two decodes the product runs cost 0.50 CPU-s; with all 46 worker processes doing
  nothing but this pair, the host decodes **26.3 photos/s**. That is the measured set B ingest rate
  (25-28 images/s, WP-1.3 and #228): photo ingest is capped by CPU JPEG decode on this host.
- DCT-domain scaling (Pillow `draft`, OpenCV `IMREAD_REDUCED_*`) saves the IDCT only: the 1/8
  decode of a photo still costs 130 ms because Huffman decoding scans the whole file.
- The Pillow path is 1.3-1.8x OpenCV's: `exif_transpose` returns a copy even when there is
  nothing to rotate, and the product later takes `np.asarray` of it for the blur.

### 5.2 GPU decoders (batch 32, 8 decoder threads unless noted; Triton image)

| Class | nvImageCodec: images/s | host CPU ms/image | GPU busy ms/image | DALI `mixed`: images/s | host ms | torchvision nvJPEG: images/s | host ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (0.27 MP) | 2,346 | 2.8 | 0.15 | 1,941 | 3.1 | 379 | 2.6 |
| OI (0.78 MP) | 1,282 | 4.9 | 0.26 | 1,176 | 5.1 | 215 | 4.6 |
| OIO (5.2 MP) | 193 | 21.6 | 1.0 | 154 | 23.5 | 22 | 45 |
| B (17.7 MP) | 35.4 | 134 | 3.9 | 31.8 | 132 | 4.5 | 221 |

"GPU busy" is the `nvidia-smi` utilisation share times wall time per image (an upper bound on the
SM time). Batch 1 and 8 on set B: 7.4 and 13.8 photos/s at the same 135-152 ms host CPU.

- **No GPU-only Huffman on this hardware.** nvImageCodec's `GPU_ONLY` backend returned no image for
  every file on the A6000 (0.7.0.49); the working backends decode the Huffman stage on the CPU
  (hybrid). GPU decode therefore cuts the host cost of a photo from 502 ms (product pair) to
  134 ms (3.7x), not to zero. The hybrid cost is about 7.5 ms per MP of CPU.
- **One decoder instance saturates at about 35 photos/s** whatever its thread count (2: 13.5,
  4: 17.9, 8: 35.4, 16: 35.9, 32: 29.6 photos/s; it uses at most 5 cores). Throughput comes from
  several decoder instances, not more threads per instance.
- torchvision's GPU decode runs images one after the other (one core, 4.5 photos/s) and is
  ruled out for throughput; it also ignores EXIF orientation (section 6.1).

### 5.3 GPU decode on the GPU (decode plus letterbox, whole frame, blur)

| Class | `nvimgcodec_full` (decode + torch kernels per image): images/s | host ms | GPU busy ms | `dali_full` (decode + letterbox + whole-frame resize in DALI): images/s | host ms | GPU busy ms |
|---|---:|---:|---:|---:|---:|---:|
| A | 446 | 6.6 | 0.37 | 1,592 | 3.6 | 0.18 |
| OI | 350 | 9.8 | 0.72 | 1,030 | 5.6 | 0.30 |
| OIO | 99 | 28.5 | 5.1 | 166 | 24.1 | 1.1 |
| B | 25.2 | 142 | 16.8 | 33.4 | 131 | 3.8 |

The torch kernels of `gpu_decode_kernels.py` are correct (section 6.4) but slow on photos: they
work in float32 on the full frame, one image at a time (13 ms of GPU per photo beyond the decode,
mostly the antialiased resize and the int32 Laplacian of 18 M pixels). DALI's fused uint8 resize
costs under 1 ms. The production kernels must meet a budget (section 9, item 2): at most 3 ms of
GPU per 18 MP photo for letterbox, whole frame and blur sums together.

### 5.4 What decode on the PE card takes from PE (contention)

PE engine driven at saturation (`pe-load`, batch 32, 6 streams: 182-183 embeddings/s alone) while
a decoder process works through set B on the same card; two repetitions:

| GPU decode | PE alone | PE while decoding | PE lost | GPU decode alone | GPU decode beside PE | PE time lost per photo |
|---|---:|---:|---:|---:|---:|---:|
| torch kernels (`nvimgcodec_full`) | 183.2 / 181.5 | 142.5 / 141.6 | -22 % | 25.1 / 25.7 photos/s | 15.2 / 15.5 | 2.7 embeddings = 14.6 ms |
| DALI (`dali_full`) | 183.2 / 181.5 | 175.1 / 170.9 | -4.5 / -5.8 % | 31.7 / 33.0 | 20.7 / 32.0 | 0.4 embeddings = 2.1 ms |

Decode on the PE card is affordable when the kernels are efficient: about 2 ms of PE time per
photo against 14.3 ms of PE work per photo at set B's 2.6 embeddings. Float32 full-frame kernels
are not (15 ms). PCIe: the JPEG upload is 10.4 MB per photo against 54 MB of RGB (5.2x less); on
COCO 168 KB against 0.85 MB.

### 5.5 GPU memory per process

| Process | MiB |
|---|---:|
| torch CUDA context | 310 |
| + torchvision decode of 1 / 8 / 16 photos held | 494 / 952 / 1,404 |
| + nvImageCodec decoder, 1 / 8 / 16 photos held | 512 / 2,000 / 2,784 |

With 32 API workers each holding a CUDA context that is 10 GB before a single frame; one decoder
service per card holds one context and a bounded frame pool.

## 6. Measurements: correctness, fallbacks, parity

### 6.1 Decoder coverage, EXIF, ICC and chroma upsampling

`gpu_decode_bench.py make-formats` derives 27 files from one set B photo; `formats` decodes each
with Pillow (the product) and the GPU decoders. Mean / max absolute difference to Pillow in grey
levels on the RGB frame; "ok" is "decoded, no Pillow reference"; "shape" is "decoded with the
wrong orientation":

| File | nvImageCodec default | nvImageCodec `fancy_upsampling=1` | DALI `mixed` | torchvision nvJPEG |
|---|---|---|---|---|
| baseline 4:2:0 q90 (also progressive, ICC sRGB) | 0.077 / 16 | **0.010 / 3** | 0.077 / 16 | 0.077 / 16 |
| 4:2:2 | 0.054 / 15 | 0.009 / 3 | 0.054 / 15 | 0.054 / 15 |
| 4:4:4 | 0.010 / 3 | 0.010 / 3 | 0.010 / 3 | 0.010 / 3 |
| quality 50 / 100 | 0.070 / 15, 0.132 / 20 | 0.005 / 3, 0.011 / 3 | as default | as default |
| odd size 1001 x 667, tiny 16 x 16, 48 MP | up to 0.41 / 14 | up to 0.023 / 3 | as default | as default |
| greyscale | 0.005 / 1 | 0.005 / 1 | 0.005 / 1 | 0.005 / 1 |
| CMYK (Adobe) | 0.006 / 1 | 0.006 / 1 | 0.006 / 1 | error |
| EXIF orientation 2, 3, 4 | correct | correct | **wrong pixels** (16-22 mean) | **wrong pixels** |
| EXIF orientation 5-8 | correct | correct | **wrong shape** | **wrong shape** |
| PNG, WebP, TIFF, BMP | identical | identical | identical | error |
| GIF | no image (None) | no image | error | error |
| truncated at 60 % | **image returned** (Pillow raises) | same | same | same |
| 1 KB of noise mid-stream | **image returned** (Pillow raises) | same | same | error |

Findings that bind the implementation:

1. **Chroma upsampling.** Every GPU decoder's default upsamples 4:2:0 and 4:2:2 chroma by
   replication; libjpeg-turbo (Pillow, OpenCV) interpolates ("fancy"). nvImageCodec with
   `options=':fancy_upsampling=1'` matches Pillow to 0.01 grey levels (max 3, the IDCT rounding).
   The GPU decode uses that option; DALI's equivalent is `jpeg_fancy_upsampling=True`.
2. **EXIF.** nvImageCodec applies the orientation (all 8 tags correct). torchvision's CUDA decode
   and DALI's decoder as configured ignore it: a GPU decode built on them must re-orient from the
   header (`gpu_decode_kernels.apply_orientation`, tested against `ImageOps.exif_transpose` for all
   8 tags). All four public sets carry orientation 1 only, so the parity runs below cannot catch an
   orientation bug: the decoder-coverage set is the gate for it.
3. **ICC.** Neither the product (Pillow decode, no colour management) nor any GPU decoder applies
   an embedded profile; 863 of the 2,000 COCO images and about 85 % of the photos carry one. No
   difference to manage; the plan keeps "ignore ICC" and says so in the decoder's docstring.
4. **Fail-open on damaged files.** The GPU decoders return an image for a truncated file and for a
   corrupted bitstream that Pillow rejects (the product then fails the image with
   `decode_failed`). The GPU decode must not turn a rejected file into an ingested one: it checks
   for the end-of-image marker before decoding and treats any decoder warning as a failure that is
   re-decoded on the CPU, so the CPU verdict stands (section 9, item 1, "fallback"). Truncation is
   the common case and the marker check catches it; a file corrupted mid-stream with an intact end
   marker is decoded with grey blocks by every GPU decoder (measured), which is the one residual
   difference to record in the docs.
5. **Formats.** Route PNG, WebP, TIFF, BMP, GIF, HEIF and CMYK, 12-bit, lossless and arithmetic
   JPEG to the CPU (the header check of `gpu_decode_lib.route`), counted per reason. Progressive
   JPEGs decode correctly on the GPU (54 of the 500 photos are progressive).

### 6.2 Parity through the stack (`parity.py`, profile `precision`, set A)

The spike of section 7.4 (nvImageCodec decode with the fancy option, GPU letterbox, `dct8` whole
frame, GPU blur sums; crops and their PE squares still on the CPU from the downloaded frame),
captured through an isolated stack with the `ref-u8fp16` engines (12,808 items, clusters trained by
the auto-label worker), against both references:

| Gate | Rule | vs `ref-u8fp16` | vs `ref-u8fp16-repeat` | noise floor (`-repeat` vs ref, #228) |
|---|---|---:|---:|---:|
| P-box `images_identical_share` | >= 0.98 | 0.869 FAIL | 0.869 FAIL | 0.994 |
| P-box-confident unmatched | 0 | 1 FAIL | 1 FAIL | 0 |
| P-item | >= 0.99 | 0.942 FAIL | 0.942 FAIL | 1.000 |
| P-pe p01 / median | >= 0.995 / 0.999 | 0.974 FAIL / 0.9997 | 0.974 FAIL / 0.9997 | 0.99995 / 1.0000 |
| P-pew p01 / median | >= 0.995 / 0.999 (parent plan for GPU decode: >= 0.99) | 0.9906 / 0.9982 FAIL (passes 0.99) | 0.9906 / 0.9982 | 0.99994 / 1.0000 |
| P-knn overlap@10 | >= 0.95 | 0.911 FAIL | 0.910 FAIL | 0.9955 |
| P-dup near-duplicate agreement | >= 0.995 | 0.9997 PASS | 0.9997 PASS | |
| P-cluster-assign | >= 0.98 | 0.988 PASS | 0.988 PASS | 0.9997 |
| P-cluster-purity drop (points) | <= 0.5 | 0.34 PASS | -0.09 PASS | 0.45 |
| P-label-vote | >= 0.97 | 0.961 FAIL | 0.961 FAIL | 0.999 |
| P-knn-precision drop (points) | <= 0.5 | 0.00 PASS | 0.01 PASS | |
| P-label | >= 0.99 | 1.000 PASS | 1.000 PASS | |

Raw summary: `parity_summary.json` of the capture (scratch). The agreement gates fail and the
task-level gates pass: kNN precision against COCO ground truth is unchanged (0.6028 against
0.6028), cluster purity moves within the noise floor, near-duplicate decisions and class labels are
unchanged.

### 6.3 Where the difference comes from (offline, same engines, per arm)

`gpu_decode_parity.py` runs every image through four arms against the product's CPU path:
`cpu_b1` (the same CPU pixels sent alone instead of in a batch of three: the engine's own noise),
`dec` (GPU decode, then the product's CPU resizes), `gpu` (GPU decode and GPU resizes). Decoder:
nvImageCodec with `fancy_upsampling=1` (the recommended one). "Same boxes" embeds the GPU crop
squares of the CPU path's boxes, isolating the crop resize from moved boxes.

| Set | arm | P-box | P-item | unmatched confident | crop rects identical | P-pe p01 | P-pe same boxes p01 | kNN overlap | P-blur pass (abs diff <= 0.02) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (2,000) | `cpu_b1` (noise) | 0.995 | 0.9995 | 0 | 0.991 | 1.000 | | 0.996 | 0.996 |
| A | `dec` | 0.887 | 0.951 | 1 | 0.822 | 0.974 | | 0.917 | 0.933 |
| A | `gpu` | 0.867 | 0.941 | 2 | 0.812 | 0.973 | 0.996 | 0.895 | 0.929 |
| B (500) | `cpu_b1` | 1.000 | 1.000 | 0 | 0.994 | 0.997 | | 0.997 | 0.998 |
| B | `dec` | 0.926 | 0.963 | 0 | 0.598 | 0.978 | | 0.942 | 0.964 |
| B | `gpu` | 0.919 | 0.966 | 0 | 0.539 | 0.979 | 0.998 | 0.929 | 0.952 |
| OIO (60) | `cpu_b1` | 1.000 | 1.000 | 0 | | 0.997 | | 0.995 | 0.990 |
| OIO | `dec` | 0.933 | 0.967 | 0 | 0.622 | 0.981 | | 0.925 | 0.911 |
| OIO | `gpu` | 0.883 | 0.967 | 0 | 0.599 | 0.972 | 0.996 | 0.921 | 0.875 |
| OI (60) | `cpu_b1` | 1.000 | 1.000 | 0 | | 1.000 | | 0.999 | 1.000 |
| OI | `dec` | 0.932 | 0.932 | 0 | 0.691 | 0.975 | | 0.933 | 0.909 |
| OI | `gpu` | 0.881 | 0.932 | 0 | 0.664 | 0.978 | 0.996 | 0.915 | 0.894 |

Decoded pixels against Pillow: mean 0.03 grey levels, max 3, on every set with nvImageCodec's fancy
option. The same run with torchvision's nvJPEG (API image, no fancy option) matched that on COCO
but differed by up to 62 grey levels on photos and 72 on the Open Images files (non-fancy chroma);
its P-box on set B was 0.912 and its same-box P-pe p01 0.977 (against 0.998 here): the decoder
choice and its chroma option matter.

1. **The decoder, not the GPU resize, moves the detector.** Decoding alone on the GPU (`dec`) costs
   most of the P-box drop (A: 0.994 to 0.883); the GPU letterbox adds little (0.883 to 0.869). The
   decoded pixels differ by 0.03 grey levels on average (max 3) on COCO, the letterbox by 0.02, and
   still 11 % of COCO images get some box that moves past IoU 0.95 or a confidence by more than 0.02:
   this detector is chaotic near its thresholds (the same finding as the detector rebuild of WP-2.1).
2. **The resizes pass the tensor gate.** GPU letterbox against Pillow: gate T passes on 100 % of
   images of every set (best shift (0,0), mean 0.02-0.11, max 4). PE crop squares from the same
   boxes: gate T passes on 99.9-100 % and P-pe p01 is 0.996-0.998 on every set.
3. **Blur moves only through the decoder.** The blur sums are bit-identical for the same pixels
   (unit tests); the `dec` and `gpu` arms have about the same P-blur pass share (A: 0.93, B: 0.95-0.96)
   because the Laplacian amplifies IDCT differences on small boxes.

### 6.4 Detector quality against COCO ground truth (`detector_quality.py`, set A)

| Arm | AP | AP50 | precision at 0.4 | recall at 0.4 | boxes moved (better / equal / worse) |
|---|---:|---:|---:|---:|---|
| `cpu` (product) | 0.4376 | 0.5701 | 0.8576 | 0.5505 | |
| `cpu_b1` (noise, first run) | 0.4376 | 0.5702 | 0.8575 | 0.5505 | 69 (30 / 15 / 24) |
| `dec` (GPU decode) | 0.4379 | 0.5703 | 0.8573 | 0.5507 | 782 (290 / 195 / 297) |
| `gpu` (GPU decode and resizes) | 0.4377 | 0.5704 | 0.8570 | 0.5508 | 842 (320 / 203 / 319) |
| val2017 only (unseen by the COCO-trained detector): `cpu` / `dec` / `gpu` | 0.4169 / 0.4171 / 0.4171 | 0.5454 / 0.5458 / 0.5459 | | | |

The boxes that move are as often better as worse: GPU decode is **different, not worse**. The
parity gates measure agreement with today's stored outputs, so they fail; accuracy does not change.

### 6.5 The resize chain: cascading against resizing from the original (R1)

| Set | whole frame, `dct8` from `F`: p01 / median | whole frame cascaded from the letterbox | crops from `F`, same boxes: p01 / median | crops cascaded from the letterbox |
|---|---|---|---|---|
| A (0.27 MP) | 0.991 / 0.998 | 0.652 / 0.817 | 0.996 / 0.9994 | 0.929 / 0.999 |
| OI (0.78 MP) | 0.991 / 0.998 | 0.815 / 0.902 | 0.996 / 0.9992 | 0.810 / 0.946 |
| OIO (5.2 MP) | 0.982 / 0.998 | 0.762 / 0.974 | 0.996 / 0.9993 | 0.616 / 0.914 |
| B (17.7 MP) | 0.984 / 0.999 | 0.945 / 0.983 | 0.998 / 0.9996 | 0.534 / 0.866 |

Cosine to the product's vectors (P-pe, P-pew). Cascading the crops from the 640 letterbox destroys
them on photos (a 300 px object in an 18 MP frame becomes 40 px); cascading the whole frame from the
letterbox lands in neither today's space nor a clean new one. So every model input is resized from
`F` (section 3). The `dct8` whole frame passes the GPU-decode gate (0.99) on frames and the 1,024 px
set and misses it on photos (p01 0.982-0.984): an emulation that averages chroma over 16 x 16 blocks
(what a 4:2:0 file's 1/8 decode does) is item 1's first refinement; if it still misses, the whole
frame on photos rides D2.

## 7. End to end, the decision, the ceiling, the size gate

### 7.1 End to end through the stack (set B photos and set A frames, `/ingest/batch`)

Isolated stack (`gpudec`, GPU 2, published v0.5.0 images, the branch's `src/` mounted, `ref-u8fp16`
engines, 3 repetitions after a warm-up project, `--skip-cluster --skip-settle`); the "spike" arms
mount a scratch copy of `src/` with one extra module that patches the ingest pixel functions (the
design of section 7.4; not committed). Foreign SM load on GPU 2 was 0.0 % in every counted run.

| Arm | API workers | Set B images/s | API CPU-s per photo | host busy % | GPU busy % | Set A images/s | API CPU-s per frame | GPU busy % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU path (`origin/main`) | 32 | 24.32 (21.72-26.26) | 1.055-1.131 | 76-79 | 36-43 | 23.51 (23.39-23.54) | 0.069 | 95-98 |
| CPU path | 16 | 26.78 (24.59-28.10) | 1.074-1.151 | 80-82 | 36-41 | | | |
| spike v1: torch GPU decode, 32 workers | 32 | failed: GPU out of memory (each worker 2-5 GB; 48 GB card full) | | | | | | |
| spike v2: float32 kernels, frame dropped after the derived tensors | 16 | 23.39 (23.00-24.38) | 0.356-0.409 | 28-33 | 70-80 | 20.23 (20.00-20.70) | 0.164 | 96-99 |
| spike v3: half-precision resize input, convolution blur | 16 | 25.48 (24.45-26.48) | 0.345-0.361 | 34-38 | 79-83 | not run | | |

Read:

- **Host CPU per photo falls 65 %** (1.10 to 0.36-0.41 CPU-s) and the host goes from 80 % to 30 %
  busy. The remaining host cost is the hybrid Huffman decode (about 134 ms per photo, section 5.2),
  the frame download and Pillow copy that the spike still makes for the crops (about 60 ms), the
  crops and their JPEG encode, documents.
- **Throughput did not rise with float32 kernels**: the GPU became the limit (70-80 % busy; the
  GPU decode costs about 15 ms of GPU per photo, section 5.4, next to 14 ms of PE). With the kernel
  budget of section 9 item 1 met (DALI-class, about 2 ms) the GPU cost per photo is about 18 ms
  (v3 cut the derived-tensor kernels from 11.3 to 8.0 ms of GPU per photo, single process, same
  pixels within 1 grey level and bit-identical blur, and gave 25.5 photos/s: still GPU-bound, still
  above the budget).
- **Frames lose.** Set A: 23.5 to 20.2 images/s (-14 %) and API CPU per frame up from 0.069 to
  0.164 s: frames are PE-bound, the spike adds GPU work and, in the API-side design, serialises a
  GPU call and its synchronisation per small image. Frames stay on the CPU path (section 7.3).

### 7.2 Against the decision rule

| Condition | Result |
|---|---|
| End-to-end images/s on set B up materially | not with the measured kernels (v2 23.4, v3 25.5 against 26.8 photos/s); the ceiling with the kernel budget met is about 1.8x on one A6000 and about 2.7x on two (section 7.5), host-bound at about 73 photos/s |
| Total CPU down | yes: -65 % API CPU per photo, -50 % host busy at a similar rate |
| Parity gates hold | no: the decoder alone fails P-box, P-item, P-pe, P-knn, P-label-vote against the CPU references (section 6.2); the difference is quantified and accuracy-neutral (AP, kNN precision, cluster purity unchanged, 6.4) |
| COCO-size frames | gain nothing and lose 14 % in the API-side design: keep them on the CPU path |

Recommendation: **GPU decode for photos only, as a per-deployment opt-in that changes the stored
numerics once (owner decision D1), built as the Triton-resident GPU decode with efficient kernels;
frames stay on the CPU path.** Not before the kernels meet the 3 ms budget: with the float32 torch
kernels the change only moves CPU to the GPU (same images/s, 65 % less CPU). Nothing is built
before D1; the first build step is item 1's kernel budget, because it is the measured binder.

### 7.3 The size gate (R3)

Measured per-image costs (sections 5.1-5.3 and WP-1.2's stage table): the CPU GPU decode of ingest
costs about 10.6 + 46.6 x MP ms of API CPU per image (decode, letterbox, whole frame, blur; fitted
on 0.27 MP frames, 23 ms, and 17.7 MP photos, 837 ms); the GPU decode about 2.5 + 7.5 x MP ms of
host CPU (hybrid Huffman) and 0.1 + 0.21 x MP ms of GPU with DALI-class kernels. Fitted on the four
classes, the GPU path's host cost is below the CPU path's from 0.77 MP up
(`gpu_decode_lib.crossover_megapixels`), and measured it is below at every class, including 0.27 MP
(2.8 against 7.4 ms for the decode alone). Host time is the wrong currency on frames, though: they are
PE-bound, so what matters is whether the host binds before the GPU does. It does when
`(60 + 10.6 + 46.6 x MP) ms x images/s > 28 cores` (60 ms of non-pixel API CPU per image, 28 cores
the API reached at its limit in WP-1.3 and #228), with images/s set by PE (181 embeddings/s per A6000
over the embeddings per image):

| Cards | Embeddings per image | PE-bound images/s | host binds above (MP) |
|---:|---:|---:|---:|
| 1 | 7.4 (COCO density) | 24.5 | 23 |
| 1 | 2.6 (photo density) | 70 | 7.1 |
| 2 | 7.4 | 49 | 10.7 |
| 2 | 2.6 | 139 | 2.8 |

Below the threshold GPU decode cannot raise images/s (the GPU binds first) and only adds GPU work;
above it the CPU path is what limits. Default `ingest_gpu_min_megapixels = 3.0` (the two-card,
photo-density row, rounded up): COCO frames (0.3 MP) and Open Images-sized frames (0.8 MP) stay on the
CPU, 12-24 MP photos go to the GPU. Caveat that the setting text must carry: a project whose images
straddle the threshold gets two decoders in one embedding space (section 6.3: P-pe p01 about 0.97
between them). For a mixed project prefer `cpu` or `gpu` for the whole project over `auto`.

### 7.4 Where the GPU decode lives

| | API process (the spike) | Triton Python-backend model (`gpu_image_prep`) | DALI ensemble |
|---|---|---|---|
| Decode once across detect and crops | natural (the frame lives in the request) | needs state between two calls: sequence batcher, correlation id per image (item 2) | no: one sample in, one out; crops after detection need the frame again (parent plan section 9) |
| GPU memory | one CUDA context and decoder per API worker: 32 workers ran the 48 GB card out of memory (spike v1); 16 workers took 12 x about 1.5 GB | one context per instance; budgeted in `config.pbtxt` like PE (`GPU_MIB_PER_INSTANCE`) | DALI instances are large and keep memory (parent plan risk 3) |
| 12 GB card (RTX 3080 Ti) | no (API GPU plus Triton's models) | yes as a secondary card with 1 instance next to PE (budget it) | no |
| Copies of pixels to the host | letterbox (1.2 MB) and squares (0.34 MB each) go back to the host for gRPC; the spike also downloads the frame for the CPU crops | none: BLS hands GPU tensors to the detector and PE through DLPack | none |
| Batching | per API worker, per request; no cross-worker batching | per instance over requests from all workers (dynamic batching across sequences) | dynamic batching |
| Failure isolation | a CUDA fault kills an API worker | a stub fault fails a request; the API falls back to the CPU path per image | as Triton |
| Multi-GPU | API pinned to `API_GPU_ID` | one instance group per listed card, as PE (#228) | as Triton |
| Risk | measured | unmeasured BLS overhead and state handling (C2, C3, C6) | detection-dependent crops impossible |

Choice: the Triton-resident model, with the API-side design as the measured fallback under the 15 %
stop rule. The work done is the same either way (same decoder, same kernels), so the spike's numbers
bound what the Triton design must reach.

### 7.5 Expected ceiling after the change (set B, photos, policy `all`)

| | One A6000 | Two A6000 |
|---|---:|---:|
| Host CPU per photo, GPU decode incl. GPU crops and crop JPEGs on the GPU (estimate from 7.1 minus the frame download and CPU crops) | about 0.33 CPU-s | about 0.33 |
| Host ceiling at 28 cores | about 85 photos/s | about 85 |
| GPU per photo: PE 2.6 x 5.5 ms + detector 1.2 + decode and GPU decode with the 3 ms budget (about 2-4) | about 19 ms | about 19 ms per card |
| GPU ceiling | about 52 photos/s | about 104 photos/s |
| Expected | **about 50 photos/s (1.9x today's 26.8)** | **about 80 photos/s, host-bound (3x)** |

Estimates from measured stage costs, not measured end to end; the two-card column also assumes the
GPU decode scales with instances the way PE did (1.91x on frames, #228). Today's CPU path cannot pass
about 26 photos/s on this host whatever the GPUs (the decode pair alone saturates 46 processes at
26.3 photos/s, section 5.1).

## 8. Batching and scaling (R2, and the owner's point c)

What is measured and stays:

- **PE.** The engine does 181-195 embeddings/s on one A6000 at every batch from 8 to 32 (5.1-5.5 ms
  per row, `perf_analyzer`, Wave 0 and #228); Triton's dynamic batcher already packs the crops of
  concurrent images into executions of mean 27.7 of 32 rows at 16 clients (WP-2.3 histogram), and a
  client-side coalescer that cut PE requests per image from 2.07 to 0.35-0.44 changed nothing
  (removed again). Instance count 2 or 3 on one card, `max_queue_delay_microseconds` 2,000 against
  15,000 and removing `preferred_batch_size` were all within 3 % (Wave 0 probe 1b). Keep
  `max_batch_size: 32`, `preferred_batch_size: [4, 8, 16, 32]`, 15 ms queue delay, one instance per
  card with no `gpus` list (data-parallel over the listed cards, #228), engine profile opt 8 / max 32
  (the per-row cost is flat from 8 to 32, so the opt shape is not a lever).
- **Detector.** Batched per request in chunks of the profile's `batch_limit` (16): 1-5 ms of GPU per
  image; not a lever.

What the GPU-resident path changes:

1. **Decode batches.** One request's images are decoded together (nvImageCodec batch decode; batch
   32 is 4.8x batch 1 on photos, 6.7x on frames). Requests from different API workers do not share a
   decode call; that is fine because one decoder instance saturates at about 35 photos/s and the
   scaling unit is the instance (below), not the batch.
2. **PE requests.** The GPU decode writes the whole-frame square and every selected crop square of
   all images in the request into one preallocated uint8 `[N, 3, 336, 336]` tensor and sends it in
   chunks of 32 rows (a photo request of 8 images is about 21 rows: one PE request instead of 16).
   The dynamic batcher keeps packing across requests exactly as today; nothing in its configuration
   changes. The whole-frame row joins the crop rows of the same request (today it is a separate
   one-row request per image, 2.07 requests per image).
3. **Scaling unit.** GPU decode instances per card: one instance saturates at about 35 photos/s of
   decode (5 host cores of hybrid Huffman), so the instance count is sized from the target photo rate:
   2 per card for the 1-card target of about 50 photos/s (section 7), 2 per card again with 2 cards.
   Frames need 1 (2,346 frames/s per instance).
4. **Back-pressure.** Unchanged at the API (`ingest_gate` admits images by decoded megabytes, WP-1.3),
   plus the GPU decode's own bound: at most `frames_in_flight` decoded frames per instance (default 16,
   about 0.9 GB of photo frames), excess requests wait in Triton's queue (`max_queue_size` 64, then
   the API gets `UNAVAILABLE` and retries the image on the CPU path, counted). The PE queue is
   unaffected because the GPU decode submits PE work only for frames it holds.
5. **Two cards.** The GPU decode runs one instance group per listed card like PE (#228); the API
   picks the card by the Triton instance it lands on (Triton's scheduler), so PE crops of an image are
   embedded on the card that decoded it only when the GPU decode calls PE itself (Triton-resident,
   stage 2). With the API-side GPU decode (stage 1) the squares travel through the host and any card's
   PE instance takes them.

## 9. Implementation plan

Tier: Sonnet implements, Opus reviews items 2 and 4 (state handling, fallback, fail-closed paths).
Every item: branch off `main`, red-first tests, a mutation check, `.venv/bin/pre-commit run
--all-files`, the offline suite, conventional commits, merge with `--no-ff`. Nothing is enabled by
default until item 6 and the owner decision D1 (section 11).

### Item 0. Owner decisions D1-D3 (blocking for item 6, not for items 1-5)

See section 11. Items 1-5 can land with the setting defaulting to the CPU path.

### Item 1. One home for the GPU decode operations

- Move the operations out of the bench into the model directory that will run them:
  `models/gpu_image_prep/1/prep_ops.py` gets `letterbox_u8`, `pe_square_u8`, `crop_u8`,
  `gray_u8`, `lap_sums_gray` (chunked), `var_from_sums`, `whole_frame_u8` (only the `dct8` mode;
  `antialias` and `area` are measured rejects and are deleted), `apply_orientation`, and from
  `gpu_decode_lib.py` the geometry (`letterbox_geometry`, `pe_square_geometry`, `crop_rect`) and the
  header router (`jpeg_header`, `RoutePolicy`, `route`). `scripts/bench/gpu_decode_kernels.py` and
  the moved parts of `gpu_decode_lib.py` are deleted in the same commit and the bench scripts import
  from the model directory (load by path as `tests/test_ocr_pipeline_rec_width.py` does); no second
  copy.
- Tests: move `tests/bench/test_gpu_decode_kernels.py` and the header/geometry tests to
  `tests/test_gpu_image_prep_ops.py` (CPU torch; the repo `.venv` has it). They already pin:
  geometry identical to `letterbox_to_square` / `resize_crop_rgb` / `crop_pil`, gate T for the two
  resizes, grey plane and Laplacian variance bit-identical to OpenCV and `crop_quality`, chunked sums
  exact, EXIF re-orientation equal to `ImageOps.exif_transpose` for all 8 tags, router reasons.
- Add the kernel budget test as a live-marked test (GPU only): an 18 MP frame through letterbox,
  whole frame and blur sums under 3 ms of GPU (section 5.3 measured 13 ms for the float32 torch
  versions). Implementation choices that meet it, in order of preference: `F.interpolate` on a
  uint8-to-half view of the frame computed once and shared by the letterbox and the 8 x 8 mean;
  a fused grey + Laplacian sum kernel (`cupy.RawKernel`, CuPy 13.6 is in the Triton image) that
  reads the RGB frame once and accumulates per-block int64 partial sums; DALI's `fn.resize`
  (`antialias=True`, `interp_type=INTERP_LINEAR`) only if its output passes gate T against
  Pillow. Bit-exactness of the blur sums is non-negotiable (test).

### Item 2. Triton model `gpu_image_prep` (decode once, state per image)

Python backend in the Triton image (nvImageCodec 0.7, torch, CuPy already there; no image change).

- `models/gpu_image_prep/config.pbtxt`: `backend: "python"`, `max_batch_size: 16`,
  `instance_group [{ count: 2, kind: KIND_GPU }]` with no `gpus` list (one group per listed card,
  as PE in #228), `parameters { key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: { string_value: "no" } }`,
  `sequence_batching { direct { } max_sequence_idle_microseconds: 30000000 control_input [ START,
  END, CORRID (UINT64) ] }` so both calls of one image reach the same instance, and a
  `# GPU_MIB_PER_INSTANCE` budget comment that `openprocessor gpu plan` reads (measure it in item 0
  of this list's probe: context plus decoder plus `frames_in_flight` frames; section 5.5 gives
  310 MiB + about 2 GiB for 8 held photos).
- Call 1 (`START`), per image: input `jpeg` (UINT8 `[-1]`); the model checks the header
  (`route`), decodes the batch of requests with one nvImageCodec call
  (`options=':fancy_upsampling=1'`, EXIF applied), keeps the frame `F` under the correlation id,
  builds the letterbox and calls the detector (BLS, DLPack, one call for the batch), builds the
  whole-frame square (`dct8`) and calls PE for it (BLS), computes the full-frame blur sums. Outputs:
  `status` (INT32: 0 decoded on the GPU, otherwise the `route` reason code; the API then runs its CPU
  path for that image), `size` (INT32 `[2]` after EXIF), `num_dets`/`det_boxes`/`det_scores`/
  `det_classes` exactly as the detector returns them plus `scale`/`pad` (so
  `decode_primary_row` is reused unchanged), `whole_embedding` (FP32 `[1024]`), `blur_sums`
  (INT64 `[3]`).
- Call 2 (`END`), per image: inputs `rects` (INT32 `[N,4]`, the `crop_rect` rectangles the API
  computed from the items it kept), `embed` (BOOL `[N]`, the embedding policy's choice),
  `crop_jpeg_max_side` (INT32, 0 = full size; WP-4.1 sets it); the model cuts the crops from `F`,
  builds the PE squares of the selected ones into one tensor, one PE BLS call in chunks of 32,
  computes the box blur sums on the grey plane, encodes the crop JPEGs with nvImageCodec, frees `F`.
  Outputs: `embeddings` (FP32 `[M,1024]`), `box_blur_sums` (INT64 `[N,3]`), `crop_jpegs` (BYTES `[N]`).
- State rules: at most `frames_in_flight` frames per instance (parameter, default 16); a `START`
  beyond it answers `status = busy` (the API decodes that image on the CPU, counted); a frame whose
  `END` never arrives is freed by an LRU sweep at the idle timeout; `START` on a reused correlation id
  frees the old frame first. Unit tests drive `execute` with a fake `pb_utils` (precedent:
  `tests/test_ocr_pipeline_rec_width.py`) for: slot reuse, overflow -> `busy`, `END` without
  `START` -> error status, fallback codes, output shapes for zero detections and zero selected crops.
- Failure isolation: a decode error or decoder `None` is per image (status code), never a failed
  batch; a CUDA error in the stub fails the request and the API retries those images on the CPU path;
  Triton restarts a dead stub instance.

### Item 3. API client and ingest integration

- `src/clients/gpu_image_prep.py`: `GpuImagePrepClient.prepare(images: list[bytes], corr_ids)` and
  `GpuImagePrepClient.finish(corr_id, rects, embed_mask, jpeg_max_side)` over the existing
  `AsyncTritonPool` (gRPC, sequence id = a 64-bit hash of `ingest_run_id` + image index; the pool
  must pass `sequence_id`, `sequence_start`, `sequence_end`).
- `src/services/curation/ingest_batch.py` `_prefill_detections`: when the setting selects the GPU
  GPU decode, call `prepare` instead of `_decode_image` + `run_primary_batch`; images with a non-zero
  status take today's path (`_decode_image`, CPU letterbox), counted in a new
  `op_ingest_decode_total{backend, reason}` counter. `ImageContext` gets `gpu_frame: GpuDecodeFrame |
  None` (correlation id, size, whole embedding, blur sums) and `pil` becomes optional; every
  consumer of `ctx.pil` (crop, blur, crop cache) branches on `ctx.gpu_frame` in one place
  (`ingest_index.index_items`).
- `ingest_index.index_items`: with a GPU decode, `embed_items` sends the selected rectangles to
  `finish` instead of `embed_crops`; the whole-frame vector comes from `prepare`; blur values are
  computed with `crop_quality.blur_ratio` from `var_from_sums` of the returned sums (same rounding
  as today); crop JPEGs from `finish` go to `write_crop_cache` (new `write_crop_cache_bytes` that
  takes encoded bytes).
- Setting: `ingest_decode` = `cpu` (default) | `gpu` | `auto` (`auto` = `gpu` for images at or
  above `ingest_gpu_min_megapixels`, default from section 7.3). Names are proposals (wave plan
  convention): the item picks the `OP_*` names, reads them once at startup into `CurationConfig`,
  and adds them to `env.template` (Curation quick-config block) and `tests/test_env_surface.py`. The embed pass (`reprocess_embed`) and dataset import stay on the CPU
  path in this item; making them use the GPU decode is item 5.
- Contract: no route or wire model changes (`make contracts-check` stays clean).

### Item 4. Fallback and fail-closed rules (tests first)

- Header router reasons (`not_jpeg`, `corrupt`, `precision`, `cmyk`, `small`, `huge`, `busy`,
  `decoder_none`, `cuda_error`) each map to the CPU path and a counter label; none is silent.
- The end-of-image marker check runs before the GPU decode; a missing marker routes to the CPU, so
  a truncated file still fails with `decode_failed` exactly as today (test with the
  `make-formats` truncated file).
- A mid-stream corrupted file with an intact marker decodes with grey blocks on the GPU and is
  rejected by the CPU (measured); record it in `docs/CURATION.md` as the one behavioural difference
  of `gpu` mode, or (D3) re-validate such files on the CPU, which costs the full CPU decode and is
  not recommended.
- CMYK goes to the CPU (nvImageCodec decodes it correctly, but Pillow's CMYK conversion is the
  reference and only 1 file in the public sets is CMYK; keep it simple).

### Item 5. Remaining decoders of the same images

The embed pass (`reprocess_embed`), `reprocess_detect`, dataset import and the crop-cache miss path
(`crop_bytes.crop_jpeg_from_disk`) decode sources again. Each gets the same GPU decode through
`GpuImagePrepClient` when `ingest_decode` is not `cpu`, so a project embedded at ingest and one
embedded later produce vectors from the same decoder (the rule of #52 and WP-2.3). The core routes
(`/detect`, `/embed`) are the parent plan's Wave 6 (stateless DALI ensembles) and stay out of this
work package.

### Item 6. Gates and benchmark, then the default

- Parity: `scripts/bench/parity.py capture` with `ingest_decode` set to `gpu` against the new reference
  of D1, profile `precision`, plus the CPU-against-GPU table of this plan re-run with
  `gpu_decode_parity.py` on sets A, B, OI and OIO (the cascade rejects removed), and
  `detector_quality.py` AP on set A (val2017 split too). Pass rules: section 7.2.
- Benchmark: `ingest_sweep_suite.py` set B `8x16` and set A `32x16`, `/ingest/batch`, 3 repetitions, both
  backends in the same session, one and two A6000s; acceptance in section 7.1.
- Default: `auto` with the measured threshold only after D1 and both gates; otherwise ship `cpu`.
- Rollback: setting `ingest_decode` back to `cpu`; the model stays loaded and unused.

## 10. Verification

Commands (scratch paths are examples; run from the repository root):

```bash
# unit tests of the probes and kernels (CPU torch)
.venv/bin/python -m pytest tests/bench/test_gpu_decode_lib.py tests/bench/test_gpu_decode_kernels.py -q --no-cov

# decoder coverage set and the decoder matrix (Triton image, one GPU)
python scripts/bench/gpu_decode_bench.py make-formats --source <one public JPEG> --out /out/formats
python scripts/bench/gpu_decode_bench.py formats --images /out/formats --out /out/formats.json

# CPU and GPU decode cost per size class (API image for cpu, Triton image for gpu)
python scripts/bench/gpu_decode_bench.py cpu --images <set> --label B --limit 60 --out /out/cpu_b.json
python scripts/bench/gpu_decode_bench.py gpu --images <set> --label B --limit 96 \
    --arms nvimgcodec:32,nvimgcodec_full:32,dali_full:32 --threads 8 --seconds 15 --out /out/gpu_b.json

# what decode takes from PE on the same card (Triton with pe_image_encoder on that card)
python scripts/bench/gpu_decode_bench.py pe-load --url 127.0.0.1:<grpc> --seconds 60 --out /out/pe.json

# CPU path against GPU decode and GPU resizing through the same engines (API image with a GPU,
# nvImageCodec mounted), then the gate table and AP
python scripts/bench/gpu_decode_parity.py capture --images <set> --url 127.0.0.1:<grpc> --out /out/par.npz
python scripts/bench/gpu_decode_parity.py compare --capture /out/par.npz --out /out/par.json --dets-dir /out/dets
.venv/bin/python scripts/bench/detector_quality.py --annotations <instances_val2017.json> \
    --annotations <instances_train2017.json> --manifest scripts/datasets/manifests/coco_bench_2000.json \
    --dets cpu=/out/dets/dets_cpu.json --dets gpu=/out/dets/dets_gpu.json --out /out/dq.json
```

## 11. Owner decisions, open questions, not measured

Owner decisions needed:

| ID | Decision | Recommendation |
|---|---|---|
| D1 | GPU decode changes the decoded pixels (IDCT rounding and, without the fancy option, chroma), and the detector turns that into different boxes on 7-13 % of images (P-box fails against the CPU path although AP is unchanged). Accept a re-baselined parity reference for `gpu` mode (a new `ref-gpudec` capture, then gate later work against it), or keep `gpu` mode out | Accept for photo projects, measured AP-neutral; keep the CPU path as the reference for existing projects; never mix the two decoders inside one project's stored vectors (setting per deployment, recorded in the image documents' existing provenance only if WP review asks for it) |
| D2 | Whole-frame vector from the 1/8 DCT emulation (cosine 0.998 median, 0.991 p01 to today's) or from a proper resize of the full frame (a new embedding space, parent plan Wave 10) | The emulation: it keeps the stored space within the GPU-decode gate; the Wave 10 decision stays separate |
| D3 | A file corrupted mid-stream with an intact end marker: accept the GPU's grey-block decode (and say so in the docs) or re-validate on the CPU | Accept and document; truncation, the common case, still fails closed |

Open questions (answer inside item 2 before building on them): C2 (does an `async` BLS
`execute` overlap calls), C3 (BLS per-call overhead for GPU tensors of 1, 10, 60 MB and the CUDA pool
size it needs), C6 (DLPack hand-off from nvImageCodec to a BLS input without a copy), sequence
batcher behaviour with the Python backend (routing by correlation id, idle eviction), the Python
stub's GPU memory per instance. Stop-and-escalate rule of the parent plan: if the Triton-resident
GPU decode loses by more than 15 % to the API-side spike design of section 7.4, build the API-side
design instead (bounded to 16 workers, frames freed after the derived tensors are built).

Not measured here:

- The Triton-resident GPU decode itself (C2, C3, C6 and the sequence batcher) and the production
  kernels against the 3 ms budget; the ceiling of section 7.1 assumes the budget is met.
- Two A6000s with GPU decode (GPU 0 was busy with another project all day) and the 12 GB card (GPU 1
  reserved): section 7.1 extrapolates the two-card number from #228's measured 1.91x on frames.
- The region stage, VLM and segmenter on crop JPEGs encoded by nvImageCodec instead of Pillow
  (P-vlm, P-sam, P-region; WP-4.1 changes the same bytes and owns those gates).
- The embed pass, dataset import and the crop-cache miss path on the GPU (item 5).
- Arithmetic-coded, 12-bit and lossless JPEGs: the tools available could not produce them; the
  router sends them to the CPU by header without a measurement.
- HEIF/HEIC: neither Pillow in the API image nor the GPU decoders read it; out of scope.
- The OI and OIO samples are 60 images each: their parity numbers carry wide intervals.

## 12. Artifacts

| What | Where |
|---|---|
| Probe tools and their tests | `scripts/bench/gpu_decode_bench.py`, `gpu_decode_parity.py`, `gpu_decode_kernels.py`, `gpu_decode_lib.py`; `tests/bench/test_gpu_decode_lib.py`, `test_gpu_decode_kernels.py` |
| Summary tables | this file and `docs/PERFORMANCE.md` ("GPU decode measurements (#217)") |
| Raw JSON, logs, parity NPZ, the spike module and its compose overrides | the run's scratch directory, not committed (raw parity captures are hundreds of MB) |
| The spike (section 7.4) | not committed; its design is section 7.4 and item 3, its numbers are in section 7 |

## 13. Build and A/B after the owner decisions (go / no-go)

Owner decisions: D1 yes (a new parity reference for the new decoder numerics, after an AP check),
D2 no (no 1/8-scale emulation: the whole-frame square is resized from the full frame, a new vector
space), D3 no silent acceptance (a truncated or damaged file is still refused). The owner then asked
for the CPU single decode first and the GPU decode built and measured properly against it.

What was built (branch `perf/gpu-decode-impl`, on top of the CPU slices):

| Slice | What | Where |
|---|---|---|
| CPU single decode | the whole-frame square is cut from the frame decoded for the detector (Pillow `BILINEAR`, shorter edge to 336, centre crop); `exif_transpose` only when the tag moves pixels | `pe_preprocess.whole_frame_rgb_from_image`, `ingest._decode_image` |
| libjpeg-turbo decode | JPEG through `simplejpeg` (the libjpeg-turbo Pillow runs, pixels identical) straight into the array the blur reads | `src/services/curation/image_decode.py` |
| GPU decode | Triton Python model `gpu_image_prep`, decoupled, one process: nvJPEG decoupled API with the GPU-Huffman backend through ctypes (16 lanes), EXIF on the GPU, fused uint8 CuPy kernels (separable Pillow-`BILINEAR` resampler for the letterbox and whole frame, plain bilinear for crop squares, exact grey + Laplacian sums), detector and PE through BLS, crop JPEGs encoded on host threads; per-image worker pool, bounded in flight | `models/gpu_image_prep/`, `src/services/curation/ingest_gpu_decode.py` |
| Routing | `OP_INGEST_GPU_DECODE` (on by default; the installer writes 0 for profiles without room), size gate `OP_INGEST_GPU_MIN_MEGAPIXELS` (3.0); the model is loaded on first use and used only when its detector input matches the ingest profile; declined files (progressive, no end-of-image marker, not a baseline JPEG, decoder error) take the CPU path, counted in `op_ingest_decode_total{route}` | `ingest_batch._prefill_gpu_decode` |

Departures from sections 7.4 and 9: nvImageCodec decodes the Huffman stage on the CPU on this card,
so the GPU decode calls nvJPEG's GPU-Huffman backend itself; there is no sequence batcher (one request
returns every product of the image, so the frame never needs to outlive the request); batching
requests inside the model put a barrier between stages and ran at 15 photos/s, per-image workers
whose BLS calls the detector's and PE's own batchers group run at 43-48 (perf_analyzer, 32 streams,
GPU 65-75 % busy). The fused kernels cost about 1.3 ms of GPU per 18 MP photo.

A/B, one isolated stack on GPU 2 (foreign SM load 0.0 % in every counted run), `/ingest/batch`, 3
repetitions at the bulk points and the latency points, median (min-max):

| Set B photos (500, 12-24 MP) | CPU path (`origin/main`) | CPU single decode | + libjpeg-turbo | GPU decode |
|---|---:|---:|---:|---:|
| detect + embed, batch 8 x 16 clients: images/s | 27.0 (25.2-27.2) | 26.8 (24.3-28.4) | 23.5 (22.2-26.6) | **35.0 (24.2-38.5)** |
| detect only (`lazy`), batch 8 x 16: images/s | 28.3 (27.6-29.4) | 31.6 (25.4-31.8) | 31.0 (29.3-31.1) | **44.0 (38.8-44.8)** |
| one client, batch 1: images/s, p50 s | 0.97, 0.98 | | 1.17, 0.81 | **2.40, 0.36** |
| 4 clients, batch 1: images/s, p50 s | 3.45, 1.07 | | 4.48, 0.85 | **8.30, 0.41** |
| host CPU-s per photo (API + Triton), bulk | 1.12 | 0.97 | 1.00 | **0.34** |
| GPU 2 busy, bulk detect + embed | 35 % | 33 % | 28 % | 55 % |

| Set A COCO frames (2,000, 0.3 MP), batch 32 x 16 | CPU path | CPU single decode | + libjpeg-turbo | GPU decode, gate 3 MP | GPU decode forced on frames (gate 0) |
|---|---:|---:|---:|---:|---:|
| detect + embed: images/s | 28.6 | 29.0 | 29.3 | 29.0 | 22.7 |
| detect only: images/s | 120.3 | 132.4 | 123.2 | 124.5 | 63.6 |

PE on the same card while the GPU decode works through set B: 223 to 213 inferences/s (-4.5 %).

Accuracy (one capture and one AP check, on the shipped configuration):

| Check | Result |
|---|---|
| Parity, `ref-ship` (GPU decode on, gate 3 MP, so COCO frames take the CPU single decode) against `ref-u8fp16`, `precision` profile | every gate passes (P-box 0.994, P-pe p01 0.9998, P-knn 0.982, P-cluster-assign 0.999, P-label-vote 0.993, kNN precision and cluster purity up 0.09 and 0.35 points) except P-pew (whole-frame p01 0.65, median 0.82): the new whole-frame space of D2, as section 3 measured. `ref-ship` is the new reference; report in `docs/benchmarks/parity/decode_once_vs_ref/` |
| AP of the GPU decode, all 2,000 COCO frames through the GPU decode against the CPU decode, same engine, confidence 0.4 | AP 0.4378 against 0.4376, AP50 0.5705 against 0.5701, recall 0.5508 against 0.5505, precision 0.8571 against 0.8576; of 9,659 matched boxes 846 moved, 317 closer to the truth and 330 farther (mean IoU change +0.0007): AP-neutral, the D1 condition. Report in `docs/benchmarks/parity/detector_quality_gpu_decode/` |

Go / no-go:

- **CPU single decode alone: no-go as a speed-up** (1.0x at bulk, under the 1.5x bar); it stays as
  the CPU path the GPU decode falls back to and the definition of the whole-frame square both paths
  share. libjpeg-turbo helps the latency mode (+21-30 %) and is noise at bulk.
- **Triton GPU decode: go, opt-in, size-gated.** Photos: 1.3x at bulk with detection and
  embedding (the rest of the pipeline and PE bind next, GPU 55 % busy), 1.55x detection only,
  2.4-2.5x for one to four clients, a third of the host CPU. Frames: the gate keeps them on the CPU
  (forced onto the GPU they lose 21-49 %), so the default gate of 3 MP stands for both modes.
- Not reached: the 1.9x estimate of section 7.5 assumed PE batching that the bulk run did not get
  (the first repetition, with cold kernel weight caches, ran at 24 photos/s); a second card, GPU
  decode of progressive files (11 % of set B goes to the CPU) and crop JPEGs on the GPU are the
  follow-ups.
