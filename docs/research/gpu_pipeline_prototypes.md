# GPU pipeline prototypes PR-0, PR-1, PR-2 (research window PW-1)

| Field | Value |
|---|---|
| Status | registered 2026-11-05, before any score; results are added below in later commits |
| Plans | [08 section 2.1 and 3](../design/v070_rebuild/08_research_prototype_track.md), [09g section 6](../design/v070_rebuild/09g_research_gpu_pipeline.md) |
| Hardware | one RTX A6000 (GPU 0), no hardware JPEG engine; driver and library versions are recorded per run in the environment JSON |
| Code | `scripts/research/gpu_pipeline/`, venv outside the repository (`requirements-gpu-research.txt`, lock beside it) |
| Protocol | tier 0 CPU smoke on 50 items, tier 1 short seeded run with today's approach as the control arm, no tier 2 without the coordinator (08 section 2.1) |

## 1. Registration

Everything in this section was fixed before any score was taken.

### 1.1 Samples, seed, order

| Set | Content | Tier-0 | Tier-1 | Order |
|---|---|---|---|---|
| A | `coco_bench_2000` pin, 640 px class | first 50 | first 200 | pin order |
| B | `commons_hires_500` pin, 12-24 MP | first 50 | first 200 | pin order |
| P | seeded sample of the owner's private photos, about 21 MP, one shared local copy, content-hash names | first 50 | first 200 | order of the seeded draw (seed 20261104) |
| OI-1024 | 200 Open Images images whose longest side is 1,024 px, drawn with seed 20261104 | - | 200 (PR-1 only) | draw order |
| EXIF fixtures | about 40 files generated from public images: orientation tags 1-8, progressive, truncated, CMYK, 4:4:4, 4:2:2, 4:2:0, restart markers | all | all (PR-1 only) | file name |

Set P is described by aggregates only in tracked text. Every dataset path is an argument.

### 1.2 PR-0, the harness and the honest GPU number

Feeds 5a-R0 and the programme stop rule: **stop optimizing when SM active is above 85 % with PE at
least 70 % of GPU ms** at the bulk point.

No arms (instrumentation). It runs the shipped algorithm out of Triton: nvJPEG GPU-Huffman lanes
(`models/gpu_image_prep/1/nvjpeg_lanes.py` imported by path), the shipped CuPy kernels, the shipped
detector and PE engines (read-only, from a checkout's model repository, loaded with the venv's
TensorRT) and the host JPEG encode of the crops, on a worker-thread pool like the shipped model.

| Check | Bar |
|---|---|
| Kernel output | the harness's letterbox, whole-frame square and crop squares are byte-identical between two runs (determinism) and pass the tensor gate against a Pillow reference: mean abs diff < 0.5 grey, max <= 8 |
| Stage split | per stage GPU ms per image (CUDA events) and wall ms; three counted repetitions after one warm-up; a stage is reported "stable" when (max - min) / median over repetitions is at most 0.25 |
| Honest GPU number | `nsys` GPU metrics sampling (SM Active, Tensor Active) and NVML utilization side by side when the counters are available; where the driver refuses profiling counters to this user (`ERR_NVGPUCTRPERM`), the SM-active leg is reported as not measurable, with the substitute measured instead: the share of wall time with at least one kernel running (union of CUDA kernel intervals from the `nsys` trace) and per-stage GPU kernel ms attributed by NVTX ranges |
| Verdict | the stop rule evaluated on the measured numbers, with the number of repetitions behind it |

### 1.3 PR-1, the decoder matrix

Feeds 5a-S4: the decoder per size class (RD-14, RD-17). Control arm: D-a, the shipped ctypes nvJPEG
`GPU_HYBRID` lanes (`nvjpeg_lanes.py`, 16 lanes).

| Arm | Setup |
|---|---|
| D-b | nvImageCodec 0.9.0.20 (the newest that DALI 2.4.0 allows: `nvidia-dali-cuda130` requires `nvidia-nvimgcodec-cu13` below 0.10, while 0.10.0.9 is the newest on PyPI; both libraries must share one venv), `HYBRID_CPU_GPU` backend, `fancy_upsampling=1`, batch 1, 8 and 64 (the best batch is the arm's value) |
| D-c | DALI `mixed` decoder, `jpeg_fancy_upsampling=True`, `hybrid_huffman_threshold` 0 and the default (the better is the arm's value) |
| coverage probe, not an arm | nvImageCodec `GPU_ONLY` on 20 images per class: the card has no hardware JPEG engine, expected to decline |
| CPU reference columns, not arms | simplejpeg (libjpeg-turbo) and Pillow with `draft` (the 5c-K4 column) |

That is two arms plus the control (the limit is three). Size classes: A (about 0.3 MP), OI-1024
(about 0.7 MP), B and P (12-24 MP), and the EXIF fixtures for coverage only.

Common conditions: 16 worker threads (or the library's own 16 CPU threads) for every arm, input
cache dropped per repetition, one untimed warm-up block of 64 images per arm, three counted
repetitions with the arm order alternating each repetition. Throughput is taken per **block of 64
files, each block timed on its own call** (200 files give 3 full blocks, the last 8 are dropped, so
9 paired blocks per arm and class; pairs share the same files). The interval is a 95 % paired
bootstrap over those pairs (5,000 draws, seed 20261104) on the ratio of medians. GPU ms per photo
comes from an `nsys` CUDA kernel trace of one repetition per arm and class. The shipped router
(`prep_geometry.route`) runs in front of every arm, so the declined set is the router's and equal by
construction (margin item 4); the raw behaviour of each arm on the declined fixtures (truncated,
progressive, CMYK) is reported separately as coverage.

**Registered margin.** An arm is a tier-2 candidate for a size class only if all hold:

1. photos/s ratio to control at least 1.15 with the lower end of the 95 % interval above 1.0 and GPU ms per photo not higher, **or** GPU ms per photo at most 0.70 of control (upper interval end below 1.0) at photos/s of at least 0.97 of control;
2. tensor gate against the Pillow decode (mean abs diff < 0.5 grey, max <= 8) on every decoded file;
3. EXIF orientation applied equal on tags 1-8 (checked on the oriented output);
4. the same set of declined files as the control (no silent acceptance).

**Early stop.** An arm stops for a class once its interval upper end on the photos/s ratio is below 1.0 after one full repetition, or it fails a gate.

### 1.4 PR-2, resize and crop operations

Feeds 5b-G2: a library that passes replaces the custom kernels. Control arm: K-a, the shipped CuPy
kernels (`prep_kernels.py`). Inputs: decoded frames (the control decoder), the four operations of
the shipped model: detector letterbox (640, Pillow bilinear with antialias), whole-frame PE square
(shorter edge 336 and centre crop), per-detection PE squares (plain bilinear, no antialias, crop
rectangles taken from the shipped detector's boxes), and the blur sums.

| Arm | Setup |
|---|---|
| K-b | CV-CUDA (0.19.0 or newest): `pillowresize` for the letterbox and the whole-frame square, `resize` (linear) for the crop squares, `laplacian` plus a gray conversion for the blur sums |
| K-c | DALI `fn.resize` with the three parity settings (bilinear with antialias, bilinear without antialias, and Pillow-compatible cropping) |

**Registered margin (5b-G2 rule).** An arm replaces a custom kernel only if the operation is
byte-identical to the control, or passes the tensor gate against the Pillow or OpenCV reference
(mean abs diff < 0.5 grey, max <= 8) and its GPU ms per photo is not higher than the control's (the
ratio's 95 % upper end at most 1.0). Box parity and PE parity (P-box, P-pe) need the engines and are
computed from the detector and PE outputs of the PR-0 harness on the arm's tensors: P-box at least
0.98 and PE cosine median at least 0.999 with p01 at least 0.995. The blur sums are exact integers
in the control; an arm passes only if its Laplacian variance equals the control's to 1e-6 relative.

### 1.5 Decisions the rows feed, and the default

The default in every row is today's behaviour: keep the shipped decoder lanes and the shipped
kernels, keep the programme stop rule as written. A prototype that does not run, stops early or
overruns is reported so and the decision is taken on today's behaviour. An arm that passes is a
"tier-2 candidate" for the coordinator; nothing is adopted here.

### 1.6 Budgets

| Prototype | Tier-1 GPU h | Stop |
|---|---:|---|
| PR-0 | 0.5 | at the budget |
| PR-1 | 1.5 | at the budget |
| PR-2 | 0.5 | at the budget |

GPU hours are wall hours of the `flock`-held commands on GPU 0, summed from the run logs.

### 1.7 Amendments made before any score

1. **PR-2 GPU ms interval.** GPU ms per photo per operation comes from `nsys` traces of two full
   passes (200 images each, NVTX range per arm and operation). "Not higher than the control" means
   the arm's median of the two passes is at most the control's, and a "lower" claim needs the
   difference to exceed the larger relative difference between that arm's own two passes. Wall ms
   per operation (median over 3 repetitions of 200 images, p90 beside it) is reported next to it as
   context: it includes host launch cost and is not the registered comparison.
2. **PR-2 fused op.** `resize_crop_convert_reformat` (CV-CUDA) is tried for the whole-frame square
   only, as a within-K-b variant (it resizes without Pillow antialiasing, so it is expected to fail
   the tensor gate on large downscales); it is reported as its own row and does not add an arm.
3. **PR-2 reference noise floor.** The detector plan is not bit-stable between two calls on the same
   canvas (observed in the tier-1 smoke before scoring), so P-box and P-item are reported for the
   control against itself (the floor) and every arm is judged against `min(0.98, floor)` for P-box.
4. **PR-0 determinism.** "Byte-identical between two runs" is evaluated on images whose detector
   rectangles are equal in both runs (the kernels get the same inputs); images where the detector
   moved a rectangle are counted and reported separately.
5. **PR-0 stage GPU ms.** The per-stage GPU ms come from the NVTX ranges of one traced repetition;
   TensorRT per-layer ranges are switched off in the harness so kernels fall into the stage that
   called them.
